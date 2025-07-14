#!/usr/bin/env python3

import pickle
import argparse
import numpy as np
import imageio
import torch
import os
from PIL import Image
from pathlib import Path
import glob
import cv2
from typing import Dict, List, Optional

# 导入现有的模块
from v2m4_trellis.utils import render_utils
from v2m4_trellis.utils.general_utils import *
# 导入新的可视化模块和预处理管道
from modules.visualization import CameraEstimationVisualizer
from modules.loss_objective import create_loss_objective
from v2m4_trellis.pipelines import TrellisImageTo3DPipeline

import sys
import trimesh
from datetime import datetime
from natsort import natsorted, ns
import utils3d
from v2m4_trellis.representations.mesh import MeshExtractResult
# 移除 rembg 导入，使用 TrellisImageTo3DPipeline 预处理
# from rembg import remove, new_session

# Import our custom loss objectives
from modules.loss_objective import create_loss_objective, list_loss_objectives

def log_progress(message):
    """Helper function to print progress with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# 移除 simple_rmbg 函数，使用 TrellisImageTo3DPipeline.preprocess_image 代替
# def simple_rmbg(image):
#     """Simple background removal using rembg"""
#     # Create rembg session
#     session = new_session()
#     # Remove background - this returns RGBA
#     result = remove(image, session=session)
#     
#     # Convert RGBA to RGB with white background
#     if result.mode == 'RGBA':
#         # Create white background
#         white_bg = Image.new('RGB', result.size, (255, 255, 255))
#         # Paste the RGBA image onto white background
#         white_bg.paste(result, mask=result.split()[-1])  # Use alpha channel as mask
#         result = white_bg
#     
#     return result

def load_glb_to_mesh_extract_result(glb_path):
    """
    Load GLB file and convert to MeshExtractResult format
    """
    mesh = trimesh.load(glb_path, process=False)
    
    # Handle different GLB structure
    if hasattr(mesh, 'geometry'):
        # Multi-geometry GLB
        geometry_key = list(mesh.geometry.keys())[0]
        actual_mesh = mesh.geometry[geometry_key]
    else:
        # Single geometry GLB
        actual_mesh = mesh
    
    # Extract vertices and faces
    vertices = torch.tensor(actual_mesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(actual_mesh.faces, dtype=torch.int64).cuda()
    
    # Extract vertex attributes (colors + normals)
    vertex_colors = None
    
    # 优先使用顶点颜色
    if hasattr(actual_mesh.visual, 'to_color'):
        try:
            vertex_colors = torch.tensor(actual_mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).cuda() / 255.0
        except:
            vertex_colors = None
    
    # 如果没有顶点颜色，尝试使用面颜色
    if vertex_colors is None and hasattr(actual_mesh.visual, 'face_colors'):
        print("🎨 Converting face colors to vertex colors...")
        face_colors = actual_mesh.visual.face_colors
        
        # 将面颜色转换为顶点颜色
        vertex_colors_np = np.zeros((len(actual_mesh.vertices), 3), dtype=np.float32)
        vertex_count = np.zeros(len(actual_mesh.vertices), dtype=np.int32)
        
        # 对每个面，将其颜色累加到对应的顶点
        for face_idx, face in enumerate(actual_mesh.faces):
            face_color = face_colors[face_idx][:3].astype(np.float32) / 255.0  # 转换为[0,1]范围
            for vertex_idx in face:
                vertex_colors_np[vertex_idx] += face_color
                vertex_count[vertex_idx] += 1
        
        # 对每个顶点，计算平均颜色
        for i in range(len(vertex_colors_np)):
            if vertex_count[i] > 0:
                vertex_colors_np[i] /= vertex_count[i]
            else:
                vertex_colors_np[i] = np.array([0.5, 0.5, 0.5])  # 默认灰色
        
        vertex_colors = torch.tensor(vertex_colors_np, dtype=torch.float32).cuda()
        print(f"✅ Converted face colors to vertex colors. Color range: [{vertex_colors.min().item():.3f}, {vertex_colors.max().item():.3f}]")
    
    # 如果还是没有颜色，使用默认灰色
    if vertex_colors is None:
        vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32).cuda() * 0.5
    
    # Compute vertex normals if not available
    vertex_normals = torch.from_numpy(actual_mesh.vertex_normals).float().cuda()
    vertex_attrs = torch.cat([vertex_colors, vertex_normals], dim=-1)
    
    # Extract texture and UV if available
    texture = None
    uv = None
    if hasattr(actual_mesh.visual, 'material') and hasattr(actual_mesh.visual.material, 'baseColorTexture'):
        texture_img = actual_mesh.visual.material.baseColorTexture
        if texture_img is not None:
            texture = torch.tensor(np.array(texture_img), dtype=torch.float32).cuda() / 255.0
            if texture.dim() == 3:
                texture = texture.flip(0)  # Flip Y axis for OpenGL convention
    
    if hasattr(actual_mesh.visual, 'uv') and actual_mesh.visual.uv is not None:
        uv = torch.tensor(actual_mesh.visual.uv, dtype=torch.float32).cuda()
    
    # Create MeshExtractResult
    mesh_result = MeshExtractResult(
        vertices=vertices,
        faces=faces,
        vertex_attrs=vertex_attrs,
        res=512,
        texture=texture,
        uv=uv
    )
    
    return mesh_result

def find_closet_camera_pos_with_custom_loss(sample, rmbg_image, loss_objective, resolution=512, bg_color=(0, 0, 0), iterations=100, params=None, return_optimize=False, prior_params=None, save_path=None, is_Hunyuan=False, use_vggt=False):
    """
    Wrapper around render_utils.find_closet_camera_pos that uses custom loss objectives.
    
    This function modifies the original implementation to use our pluggable loss system.
    """
    # If we already have params, just use the original function
    if params is not None:
        return render_utils.find_closet_camera_pos(
            sample, rmbg_image, resolution, bg_color, iterations, params, 
            return_optimize, prior_params, save_path, is_Hunyuan, use_vggt
        )
    
    # Import necessary modules (copied from original function)
    import torch.nn.functional as F
    from tqdm import tqdm
    from v2m4_trellis.renderers import MeshRenderer
    from v2m4_trellis.utils.render_utils import (
        particle_swarm_optimization, optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics,
        batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics
    )
    
    fov = 40
    device = "cuda"
    
    # Setup renderer
    options = {'resolution': resolution, 'bg_color': bg_color}
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get('resolution', 512)
    renderer.rendering_options.near = options.get('near', 1)
    renderer.rendering_options.far = options.get('far', 100)
    renderer.rendering_options.ssaa = options.get('ssaa', 2)
    
    # Prepare target image and mask
    rmbg_image = torch.tensor(np.array(rmbg_image)).float().cuda().permute(2, 0, 1) / 255
    
    # Resize rmbg_image to match rendering resolution if needed
    if rmbg_image.shape[-1] != resolution:
        print(f"🔧 Resizing rmbg_image from {rmbg_image.shape[-1]} to {resolution}")
        rmbg_image = F.interpolate(rmbg_image.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(0)

    # Get foreground mask from the rmbg image (which is not black)
    target_mask = (rmbg_image.sum(dim=0) > 0).float()

    # Batch version of fitness function using custom loss objective
    @torch.no_grad()
    def fitness_batch_custom(params, renderer, sub_batch_size=3, return_more=False):
        '''
        Custom fitness function that uses our pluggable loss objective system.
        '''
        params = torch.tensor(params, dtype=torch.float32).cuda()
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params.chunk(6, dim=1)
        lookat = torch.cat([lookat_x, lookat_y, lookat_z], dim=1)
        extr, intr = batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaw, pitch, r, fov, lookat
        )
        losses = []
        renderings = []
        pts3d = []
        
        for i in range(0, params.shape[0], sub_batch_size):
            sub_extr = extr[i:i+sub_batch_size]
            sub_intr = intr[i:i+sub_batch_size]
            current_batch_size = sub_extr.shape[0]
            
            res = renderer.render_batch(sample, sub_extr, sub_intr, return_types=["mask", "color"])
            rendering = torch.clip(res['color'], 0., 1.)  # [current_batch_size, 3, 512, 512]
            
            # 为每个批次中的每个项目分别计算损失
            for j in range(current_batch_size):
                single_rendering = rendering[j]  # [3, 512, 512]
                single_mask = res['mask'][j]     # [1, 512, 512]
                
                # 确保single_mask是[512, 512]的形状
                if single_mask.dim() == 3 and single_mask.shape[0] == 1:
                    single_mask = single_mask.squeeze(0)  # [1, 512, 512] -> [512, 512]
                
                # 验证single_rendering的形状
                if single_rendering.dim() != 3 or single_rendering.shape[0] != 3:
                    print(f"⚠️  Warning: Unexpected rendering shape {single_rendering.shape} for item {j}, expected [3, 512, 512]")
                    # 添加虚拟数据以保持批次大小一致
                    dummy_rendering = torch.ones((3, 512, 512), device=device, dtype=torch.float32) * 0.5
                    renderings.append(dummy_rendering.detach().cpu())
                    losses.append(torch.tensor(1.0, device=device))
                    continue
                
                # Use custom loss objective
                loss = loss_objective.compute_loss(
                    rendered_image=single_rendering,
                    target_image=rmbg_image,
                    rendered_mask=single_mask,
                    target_mask=target_mask
                )
                
                losses.append(loss)
                renderings.append(single_rendering.detach().cpu())

            if return_more:
                res = renderer.render_batch(sample, sub_extr, sub_intr, return_types=["points3Dpos"])
                pts3d.append(res['points3Dpos'].detach().cpu())
        
        renderings = torch.stack(renderings)
        losses = torch.stack(losses).squeeze()  # Use stack and squeeze for proper shape
        
        if return_more:
            return losses, renderings, extr, intr, torch.cat(pts3d)
        return losses, renderings
    
    print(f"🎯 Using loss objective: {loss_objective.get_description()}")
    
    # Coarse optimization with PSO using custom loss
    coarse_params = particle_swarm_optimization(
        fitness_batch_custom, None, num_particles=100, init_samples=1000, rmbg_image=rmbg_image, renderer=renderer, 
        prior_params=prior_params, save_path=save_path, is_Hunyuan=is_Hunyuan, use_vggt=use_vggt
    )
    yaw, pitch, r, lookat_x, lookat_y, lookat_z = coarse_params

    # Fine-tune with gradient descent (using simplified mask loss for stability)
    yaw = torch.nn.Parameter(torch.tensor([yaw], dtype=torch.float32))
    pitch = torch.nn.Parameter(torch.tensor([pitch], dtype=torch.float32))
    r = torch.nn.Parameter(torch.tensor([r], dtype=torch.float32))
    lookat = torch.nn.Parameter(torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32))

    optimizer = torch.optim.Adam([yaw, pitch, lookat, r], lr=0.01)
    
    # ✅ 修正后的实现
    par = tqdm(range(300), desc=f'Fine-tuning with {loss_objective.get_name()}', disable=False)
    for iter in par:
        extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat)
        res = renderer.render(sample, extr, intr, return_types=["mask", "color"])  # 需要color

        # 使用自定义损失函数进行梯度下降 ✅
        rendered_image = res['color']
        rendered_mask = res['mask'].squeeze(0) if res['mask'].dim() == 3 else res['mask']
        
        # 处理可能的形状不匹配
        if rendered_mask.dim() == 3 and rendered_mask.shape[0] == 1:
            rendered_mask = rendered_mask.squeeze(0)
        
        # 使用自定义损失函数！
        loss = loss_objective.compute_loss(
            rendered_image=rendered_image,
            target_image=rmbg_image,
            rendered_mask=rendered_mask,
            target_mask=target_mask
        )

        par.set_postfix({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()  # 现在使用可微分的自定义损失！
        optimizer.step()

    # Get the final rendering
    extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat)
    res = renderer.render(sample, extr, intr)
    ret = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    
    return ret, torch.stack([yaw[0], pitch[0], r[0], lookat[0], lookat[1], lookat[2]]).detach().squeeze().cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description='Camera Search and Mesh Re-Pose from GLB files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing GLB files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--source_images_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--use_vggt', action='store_true', help='Use VGGT for camera search')
    parser.add_argument('--model', type=str, default='Hunyuan', choices=['TRELLIS', 'Hunyuan', 'TripoSG', 'Craftsman'])
    
    # Loss objective arguments
    parser.add_argument('--loss_type', type=str, default='dreamsim', 
                       choices=['dreamsim', 'lpips', 'l1l2', 'ssim', 'hybrid', 'mask_only'],
                       help='Type of loss objective to use')
    parser.add_argument('--list_losses', action='store_true', 
                       help='List all available loss objectives and exit')
    
    # Sample limiting argument
    parser.add_argument('--max_samples', type=int, default=3, 
                       help='Maximum number of samples to process (default: 3, -1 for all)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # List available loss objectives if requested
    if args.list_losses:
        list_loss_objectives()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化器实例
    visualizer = CameraEstimationVisualizer(figsize=(15, 10))
    
    log_progress("🚀 Starting Camera Search and Mesh Re-Pose Pipeline")
    log_progress(f"📁 Input directory: {args.input_dir}")
    log_progress(f"📁 Output directory: {args.output_dir}")
    log_progress(f"📁 Source images directory: {args.source_images_dir}")
    log_progress(f"🎯 Loss objective: {args.loss_type}")
    
    # Create loss objective
    loss_objective = create_loss_objective(args.loss_type)
    log_progress(f"🎯 Loss description: {loss_objective.get_description()}")
    
    # Get all GLB files
    glb_files = [f for f in os.listdir(args.input_dir) if f.endswith('_texture_consistency_sample.glb')]
    glb_files = natsorted(glb_files, alg=ns.PATH)
    
    # Apply sample limiting
    total_files = len(glb_files)
    if args.max_samples > 0:
        glb_files = glb_files[:args.max_samples]
        log_progress(f"📦 Found {total_files} GLB files, limited to first {args.max_samples}")
    else:
        log_progress(f"📦 Found {len(glb_files)} GLB files (processing all)")
    
    log_progress(f"📦 Will process {len(glb_files)} GLB files")
    
    # Get corresponding source images
    img_files = [f for f in os.listdir(args.source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files = natsorted(img_files, alg=ns.PATH)
    
    log_progress(f"📸 Found {len(img_files)} source images")
    
    # Process each GLB file
    outputs_list = []
    base_name_list = []
    extrinsics_list = []
    
    for i, glb_file in enumerate(glb_files):
        log_progress(f"🔄 Processing {i+1}/{len(glb_files)}: {glb_file}")
        
        # Load GLB file
        glb_path = os.path.join(args.input_dir, glb_file)
        mesh_result = load_glb_to_mesh_extract_result(glb_path)
        
        # Get base name (extract frame number from filename)
        base_name = glb_file.replace('_texture_consistency_sample.glb', '')
        base_name_list.append(base_name)
        
        # Find corresponding source image
        img_file = None
        for img in img_files:
            img_base = img.split('.')[0]
            if base_name in img or img_base == base_name:
                img_file = img
                break
        
        if img_file is None:
            log_progress(f"⚠️ Warning: No corresponding image found for {base_name}, skipping...")
            continue
        
        # Load source image
        img_path = os.path.join(args.source_images_dir, img_file)
        image = Image.open(img_path)
        
        # 使用与 main_original.py 一致的预处理方法
        if args.model == "Hunyuan":
            log_progress("📸 Starting image preprocessing...")
            cropped_image, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_rgba=True)
            # 保存预处理图像
            rmbg_save_path = os.path.join(args.output_dir, f"{base_name}_rmbg.png")
            cropped_save_path = os.path.join(args.output_dir, f"{base_name}_cropped.png")
            rmbg_image.save(rmbg_save_path)
            cropped_image.save(cropped_save_path)
            log_progress("📸 Image preprocessing completed")
        elif args.model in ["TripoSG", "Craftsman"]:
            log_progress("📸 Starting image preprocessing...")
            cropped_image, rmbg_image_rgba, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_all_rbga=True)
            # 保存预处理图像
            rmbg_save_path = os.path.join(args.output_dir, f"{base_name}_rmbg.png")
            cropped_save_path = os.path.join(args.output_dir, f"{base_name}_cropped.png")
            rmbg_image.save(rmbg_save_path)
            cropped_image.save(cropped_save_path)
            log_progress("📸 Image preprocessing completed")
        else:  # TRELLIS
            log_progress("📸 Starting image preprocessing...")
            cropped_image, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image)
            # 保存预处理图像
            rmbg_save_path = os.path.join(args.output_dir, f"{base_name}_rmbg.png")
            cropped_save_path = os.path.join(args.output_dir, f"{base_name}_cropped.png")
            rmbg_image.save(rmbg_save_path)
            cropped_image.save(cropped_save_path)
            log_progress("📸 Image preprocessing completed")
        
        # Create outputs dictionary
        outputs = {
            'mesh': [mesh_result],
            'mesh_genTex': [mesh_result]  # For non-TRELLIS models
        }
        
        log_progress("📷 Starting Camera Search and Mesh Re-Pose...")
        
        # Camera Search and Mesh Re-Pose with custom loss objective
        rend_img, params = find_closet_camera_pos_with_custom_loss(
            outputs['mesh'][0], 
            rmbg_image,
            loss_objective,
            save_path=os.path.join(args.output_dir, base_name),
            is_Hunyuan=(args.model in ["Hunyuan", "TripoSG"]),
            use_vggt=args.use_vggt
        )
        
        # Save alignment image
        mesh_align_path = os.path.join(args.output_dir, f"{base_name}_sample_mesh_align.png")
        imageio.imsave(mesh_align_path, rend_img)
        
        # Initialize frame data for visualization
        frame_data = {
            'frame_id': base_name,
            'original': img_path,
            'rmbg': rmbg_save_path,
            'cropped': cropped_save_path,  # 添加裁剪后的图像路径
            'large_sampling': os.path.join(args.output_dir, f"{base_name}_1_after_large_sampling.png"),
            'dust3r': os.path.join(args.output_dir, f"{base_name}_2_after_dust3r.png"),
            'pso': os.path.join(args.output_dir, f"{base_name}_3_after_PSO.png"),
            'final_align': mesh_align_path
        }
        
        # Handle different model types
        if args.model != "TRELLIS":
            rend_img, params = find_closet_camera_pos_with_custom_loss(
                outputs['mesh_genTex'][0], 
                rmbg_image,
                loss_objective,
                params=params, 
                use_vggt=args.use_vggt
            )
            gentex_align_path = os.path.join(args.output_dir, f"{base_name}_sample_genTex_align.png")
            imageio.imsave(gentex_align_path, rend_img)
            # Update final align path to genTex version
            frame_data['final_align'] = gentex_align_path
        
        log_progress("✅ Camera search and mesh re-pose completed")
        
        # Re-canonicalization of the Mesh
        log_progress("🔄 Re-canonicalization of the Mesh...")
        
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        yaw = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch = torch.tensor([pitch], dtype=torch.float32).cuda()
        r = torch.tensor([r], dtype=torch.float32).cuda()
        lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
        
        # Get the extrinsics from the camera parameters
        orig = torch.stack([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).squeeze() * r
        extr = utils3d.torch.extrinsics_look_at(orig, lookat, torch.tensor([0, 0, 1]).float().cuda())
        extr = extr.unsqueeze(0)
        
        # Transform mesh vertices
        vertices = outputs['mesh'][0].vertices.unsqueeze(0)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2)).squeeze()
        
        # Replace the vertices of the mesh
        outputs['mesh'][0].vertices = vertices_camera[:, :3]
        
        # Transform mesh_genTex if different from mesh
        if outputs['mesh_genTex'][0] is not outputs['mesh'][0]:
            vertices_gentex = outputs['mesh_genTex'][0].vertices.unsqueeze(0)
            vertices_gentex_homo = torch.cat([vertices_gentex, torch.ones_like(vertices_gentex[..., :1])], dim=-1)
            vertices_gentex_camera = torch.bmm(vertices_gentex_homo, extr.transpose(-1, -2)).squeeze()
            outputs['mesh_genTex'][0].vertices = vertices_gentex_camera[:, :3]
        
        # Save re-canonicalized mesh
        mesh_canonical = trimesh.Trimesh(
            vertices=outputs['mesh'][0].vertices.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            faces=outputs['mesh'][0].faces.cpu().numpy(),
            vertex_colors=np.clip(outputs['mesh'][0].vertex_attrs.cpu().numpy()[:, :3] * 255, 0, 255).astype(np.uint8),
            process=False
        )
        canonical_path = os.path.join(args.output_dir, f"{base_name}_re-canonicalization_sample.glb")
        mesh_canonical.export(canonical_path)
        
        log_progress("✅ Re-canonicalization completed")
        
        # 创建可视化
        log_progress("🎨 Creating visualizations...")
        
        try:
            # 只创建完整对比网格
            grid_path = os.path.join(args.output_dir, f"{base_name}_comparison_grid.png")
            visualizer.create_frame_comparison_grid(frame_data, grid_path)
            log_progress(f"📊 Created comparison grid: {grid_path}")
            
            # 清理中间可视化文件
            intermediate_files = [
                os.path.join(args.output_dir, f"{base_name}_1_after_large_sampling.png"),
                os.path.join(args.output_dir, f"{base_name}_2_after_dust3r.png"),
                os.path.join(args.output_dir, f"{base_name}_3_after_PSO.png")
            ]
            for temp_file in intermediate_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    log_progress(f"🗑️ Cleaned up intermediate file: {os.path.basename(temp_file)}")
            
        except Exception as e:
            log_progress(f"⚠️ Warning: Visualization failed for {base_name}: {e}")
        
        # Store results
        outputs_list.append(outputs)
        extrinsics_list.append(extr)
        
        log_progress(f"✅ Completed processing {base_name}")
    
    # Save results
    import dill
    with open(os.path.join(args.output_dir, "outputs_list.pkl"), 'wb') as f:
        dill.dump(outputs_list, f)
    with open(os.path.join(args.output_dir, "base_name_list.pkl"), 'wb') as f:
        dill.dump(base_name_list, f)
    with open(os.path.join(args.output_dir, "extrinsics_list.pkl"), 'wb') as f:
        dill.dump(extrinsics_list, f)
    
    log_progress("🎉 Camera Search and Mesh Re-Pose Pipeline completed!")
    log_progress(f"📊 Processed {len(outputs_list)} meshes successfully")
    log_progress(f"📁 Results saved to: {args.output_dir}")
    log_progress(f"🎯 Loss objective used: {loss_objective.get_description()}")
    log_progress("🎨 Comparison grid visualizations created for all processed frames")

if __name__ == "__main__":
    main()
