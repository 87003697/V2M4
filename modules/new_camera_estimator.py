"""
新的相机估计器 - 基于SimpleCamEstimate架构，兼容V2M4 API
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import imageio

from .camera_pose import CameraPose, CameraPoseConverter
from .pso_optimizer import PSOOptimizer
from .camera_utils import (
    preprocess_image, batch_render_and_compare, generate_sphere_samples, 
    generate_dust3r_candidates, cleanup_gpu_memory, save_pose_visualization,
    filter_similar_poses
)


class NewCameraEstimator:
    """新的相机估计器 - 完全兼容原始API"""
    
    def __init__(self, loss_objective, renderer, device: str = "cuda"):
        self.loss_objective = loss_objective
        self.renderer = renderer
        self.device = device
        
        # 优化器配置
        self.pso_optimizer = PSOOptimizer(
            num_particles=100, 
            max_iterations=50,
            w=0.9,
            c1=2.0,
            c2=2.0
        )
        
        # 渲染参数
        self.fov = 40
        self.resolution = 512
        
        # 采样参数
        self.initial_samples = 1000
        self.top_candidates = 100
        
    def find_camera_pose(self, mesh_sample, target_image: Image.Image, 
                        resolution: int = 512, bg_color: tuple = (0, 0, 0), 
                        iterations: int = 100, params: Optional[np.ndarray] = None,
                        return_optimize: bool = False, prior_params: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None, is_Hunyuan: bool = False, 
                        use_vggt: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        主要接口 - 完全兼容原始API的相机姿态估计
        
        返回：
        - rendered_image: 渲染结果 (numpy array)
        - camera_params: 相机参数 (numpy array: [yaw, pitch, r, lookat_x, lookat_y, lookat_z])
        """
        
        print(f"🎯 Using new camera estimator with {self.loss_objective.get_name()} loss")
        
        # 如果已经有参数，直接使用
        if params is not None:
            return self._render_with_params(mesh_sample, target_image, params)
        
        # 预处理目标图像
        target_tensor = torch.tensor(np.array(target_image)).float().to(self.device).permute(2, 0, 1) / 255
        if target_tensor.shape[-1] != resolution:
            target_tensor = F.interpolate(target_tensor.unsqueeze(0), size=(resolution, resolution), 
                                        mode='bilinear', align_corners=False).squeeze(0)
        
        # 获取前景mask
        target_mask = (target_tensor.sum(dim=0) > 0).float()
        
        # 第一阶段：大规模采样
        print("📊 Phase 1: Large-scale sampling...")
        best_pose, best_score = self._large_scale_sampling(
            mesh_sample, target_tensor, target_mask, save_path
        )
        
        # 第二阶段：DUSt3R估计（简化版）
        print("🔮 Phase 2: DUSt3R estimation...")
        dust3r_pose = self._dust3r_estimation(
            mesh_sample, target_tensor, target_mask, best_pose, save_path
        )
        
        # 第三阶段：PSO优化
        print("🚀 Phase 3: PSO optimization...")
        pso_pose = self._pso_optimization(
            mesh_sample, target_tensor, target_mask, dust3r_pose, save_path
        )
        
        # 第四阶段：梯度精化
        print("🎯 Phase 4: Gradient refinement...")
        refined_pose = self._gradient_refinement(
            mesh_sample, target_tensor, target_mask, pso_pose
        )
        
        # 最终渲染
        final_params = CameraPoseConverter.to_v2m4_format(refined_pose)
        rendered_image, _ = self._render_with_params(mesh_sample, target_image, final_params)
        
        # 清理内存
        cleanup_gpu_memory()
        
        print("✅ New camera estimator completed")
        return rendered_image, final_params
    
    def _large_scale_sampling(self, mesh_sample, target_tensor: torch.Tensor, 
                            target_mask: torch.Tensor, save_path: Optional[str] = None) -> Tuple[CameraPose, float]:
        """大规模采样阶段"""
        # 生成球面采样
        sphere_samples = generate_sphere_samples(num_samples=self.initial_samples)
        
        # 批量评估函数
        def batch_objective(poses_batch):
            return batch_render_and_compare(
                mesh_sample, poses_batch, target_tensor, target_mask, 
                self.loss_objective, self.renderer, sub_batch_size=8
            )
        
        # 评估所有样本
        print("🔍 Evaluating initial samples...")
        all_scores = []
        batch_size = 50  # 每批评估50个样本
        
        for i in tqdm(range(0, len(sphere_samples), batch_size), desc="Sampling"):
            batch_poses = sphere_samples[i:i+batch_size]
            batch_scores = batch_objective(batch_poses)
            all_scores.extend(batch_scores)
        
        # 选择最佳样本
        best_idx = np.argmin(all_scores)
        best_pose = sphere_samples[best_idx]
        best_score = all_scores[best_idx]
        
        print(f"📊 Best sampling score: {best_score:.4f}")
        
        # 保存采样结果
        if save_path:
            sample_params = CameraPoseConverter.to_v2m4_format(best_pose)
            sample_img, _ = self._render_with_params(mesh_sample, None, sample_params)
            imageio.imsave(f"{save_path}_1_after_large_sampling.png", sample_img)
        
        return best_pose, best_score
    
    def _dust3r_estimation(self, mesh_sample, target_tensor: torch.Tensor, 
                         target_mask: torch.Tensor, initial_pose: CameraPose,
                         save_path: Optional[str] = None) -> CameraPose:
        """DUSt3R估计阶段（简化版）"""
        # 这里简化实现，实际可以集成真正的DUSt3R
        # 当前使用启发式方法生成候选姿态
        
        dust3r_candidates = generate_dust3r_candidates(target_tensor, num_candidates=20)
        
        # 添加初始最佳姿态
        dust3r_candidates.append(initial_pose)
        
        # 批量评估
        def batch_objective(poses_batch):
            return batch_render_and_compare(
                mesh_sample, poses_batch, target_tensor, target_mask, 
                self.loss_objective, self.renderer, sub_batch_size=8
            )
        
        scores = batch_objective(dust3r_candidates)
        best_idx = np.argmin(scores)
        dust3r_pose = dust3r_candidates[best_idx]
        
        print(f"🔮 DUSt3R estimation score: {scores[best_idx]:.4f}")
        
        # 保存DUSt3R结果
        if save_path:
            dust3r_params = CameraPoseConverter.to_v2m4_format(dust3r_pose)
            dust3r_img, _ = self._render_with_params(mesh_sample, None, dust3r_params)
            imageio.imsave(f"{save_path}_2_after_dust3r.png", dust3r_img)
        
        return dust3r_pose
    
    def _pso_optimization(self, mesh_sample, target_tensor: torch.Tensor, 
                        target_mask: torch.Tensor, initial_pose: CameraPose,
                        save_path: Optional[str] = None) -> CameraPose:
        """PSO优化阶段"""
        # 准备候选姿态
        pso_candidates = generate_sphere_samples(num_samples=self.top_candidates)
        
        # 添加初始姿态
        pso_candidates.append(initial_pose)
        
        # 过滤相似姿态
        pso_candidates = filter_similar_poses(pso_candidates, threshold=5.0)
        
        # 批量评估函数
        def batch_objective(poses_batch):
            return batch_render_and_compare(
                mesh_sample, poses_batch, target_tensor, target_mask, 
                self.loss_objective, self.renderer, sub_batch_size=8
            )
        
        # 运行PSO优化
        best_pose = self.pso_optimizer.optimize_batch(
            batch_objective, pso_candidates
        )
        
        # 保存PSO结果
        if save_path:
            pso_params = CameraPoseConverter.to_v2m4_format(best_pose)
            pso_img, _ = self._render_with_params(mesh_sample, None, pso_params)
            imageio.imsave(f"{save_path}_3_after_PSO.png", pso_img)
        
        return best_pose
    
    def _gradient_refinement(self, mesh_sample, target_tensor: torch.Tensor, 
                           target_mask: torch.Tensor, initial_pose: CameraPose) -> CameraPose:
        """梯度优化细化"""
        from v2m4_trellis.utils.render_utils import optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics
        
        # 转换为可优化参数
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = initial_pose.to_yaw_pitch_params()
        
        yaw = torch.nn.Parameter(torch.tensor([yaw], dtype=torch.float32).cuda())
        pitch = torch.nn.Parameter(torch.tensor([pitch], dtype=torch.float32).cuda())
        r = torch.nn.Parameter(torch.tensor([r], dtype=torch.float32).cuda())
        lookat = torch.nn.Parameter(torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda())
        
        optimizer = torch.optim.Adam([yaw, pitch, lookat, r], lr=0.01)
        
        # 梯度优化
        best_loss = float('inf')
        best_params = None
        
        with tqdm(range(300), desc='Gradient refinement', disable=False) as pbar:
            for iteration in pbar:
                extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, self.fov, lookat)
                res = self.renderer.render(mesh_sample, extr, intr, return_types=["mask", "color"])
                
                rendered_image = res['color']
                rendered_mask = res['mask'].squeeze(0) if res['mask'].dim() == 3 else res['mask']
                
                # 计算损失
                loss = self.loss_objective.compute_loss(
                    rendered_image=rendered_image,
                    target_image=target_tensor,
                    rendered_mask=rendered_mask,
                    target_mask=target_mask
                )
                
                # 记录最佳参数
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = (yaw[0].item(), pitch[0].item(), r[0].item(),
                                 lookat[0].item(), lookat[1].item(), lookat[2].item())
                
                pbar.set_postfix({'loss': loss.item(), 'best': best_loss})
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 返回最佳参数对应的姿态
        if best_params is not None:
            return CameraPose.from_yaw_pitch_params(*best_params)
        else:
            return CameraPose.from_yaw_pitch_params(
                yaw[0].item(), pitch[0].item(), r[0].item(),
                lookat[0].item(), lookat[1].item(), lookat[2].item()
            )
    
    def _render_with_params(self, mesh_sample, target_image: Optional[Image.Image], 
                          params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用给定参数渲染"""
        from v2m4_trellis.utils.render_utils import optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics
        
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        yaw = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch = torch.tensor([pitch], dtype=torch.float32).cuda()
        r = torch.tensor([r], dtype=torch.float32).cuda()
        lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
        
        extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, self.fov, lookat)
        res = self.renderer.render(mesh_sample, extr, intr)
        
        rendered_image = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        return rendered_image, params 