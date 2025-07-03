import torch
try:
    import kaolin as kal
    import nvdiffrast.torch as dr
except :
    print("Kaolin and nvdiffrast are not installed. Please install them to use the mesh renderer.")
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F
import numpy as np


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
        """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device

    def deepcopy(self):
        """
        Create a deepcopy of the renderer.

        Returns:
            MeshRenderer: Deepcopy of the renderer.
        """
        return MeshRenderer(self.rendering_options, self.device)
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth", "color", "texture"]
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, rast_db = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa), grad_db=True)
        
        out_dict = edict()
        for type in return_types:
            img = None
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # normalize norm pictures
                img = (img + 1) / 2
                mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                img = torch.where(mask > 0, img, torch.ones_like(img))
            elif type == "normal_map" :
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "texture":
                try:
                    uv_map, uv_map_dr = dr.interpolate(mesh.uv, rast, faces_int, rast_db, diff_attrs='all')
                    img = dr.texture(mesh.texture.unsqueeze(0), uv_map, uv_map_dr)
                    # using mask to filter out the texture, set the background to pure white
                    mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                    img = torch.where(mask > 0, img, torch.ones_like(img))
                except Exception as e:
                    print(e)
                    continue

            if ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze(0)
                if type in ["mask", "depth"] and img.shape[0] == 1:
                    img = img.squeeze(0)
            else:
                img = img.permute(0, 3, 1, 2).squeeze(0)
                if type in ["mask", "depth"] and img.shape[0] == 1:
                    img = img.squeeze(0)
            out_dict[type] = img

        return out_dict

    # Allow for batch differentiable rendering
    def render_batch(
        self,
        mesh: MeshExtractResult,
        extrinsics_batch: torch.Tensor,
        intrinsics_batch: torch.Tensor,
        return_types=["mask", "normal", "depth"],
        params={},
        return_rast_vertices=False,
    ) -> list:
        """
        Render the mesh for a batch of camera extrinsics and intrinsics.

        Args:
            mesh: MeshExtractResult object representing the mesh.
            extrinsics_batch (torch.Tensor): [B, 4, 4] batch of camera extrinsics.
            intrinsics_batch (torch.Tensor): [B, 3, 3] batch of camera intrinsics.
            return_types (list): List of return types; can include "mask", "depth", "normal_map", "normal", "color".

        Returns:
            list[edict]: A list of results, one edict per batch item, containing the requested image types.
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        batch_size = extrinsics_batch.shape[0]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((batch_size, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_list = [{k: default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types} for _ in range(batch_size)]
            return ret_list

        perspective_batch = torch.stack([intrinsics_to_projection(intrinsics, near, far) for intrinsics in intrinsics_batch])
        full_proj_batch = torch.bmm(perspective_batch, extrinsics_batch)

        vertices = mesh.vertices.unsqueeze(0).expand(batch_size, -1, -1)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera_batch = torch.bmm(vertices_homo, extrinsics_batch.transpose(-1, -2))
        vertices_clip_batch = torch.bmm(vertices_homo, full_proj_batch.transpose(-1, -2))
        
        # Fix: Ensure faces are int32 for nvdiffrast
        faces_int = mesh.faces.int()
        
        # Safety check: validate inputs
        if torch.isnan(vertices_clip_batch).any() or torch.isinf(vertices_clip_batch).any():
            print("⚠️  Warning: NaN or Inf in vertices_clip_batch")
            
        # Check for reasonable vertex values
        if vertices_clip_batch.abs().max() > 1e6:
            print(f"⚠️  Warning: Very large vertex values: {vertices_clip_batch.abs().max()}")

        try:
            rast_batch, _ = dr.rasterize(self.glctx, vertices_clip_batch, faces_int, (resolution * ssaa, resolution * ssaa))
        except RuntimeError as e:
            if "Cuda error: 700" in str(e):
                print(f"⚠️  CUDA error 700 encountered with batch_size={batch_size}, attempting fallback...")
                # Fallback: process each item individually
                individual_results = []
                for i in range(batch_size):
                    try:
                        single_vertices = vertices_clip_batch[i:i+1]
                        single_rast, _ = dr.rasterize(self.glctx, single_vertices, faces_int, (resolution * ssaa, resolution * ssaa))
                        individual_results.append(single_rast)
                    except RuntimeError:
                        print(f"⚠️  Skipping problematic item {i} in batch")
                        # Create a dummy result
                        dummy_rast = torch.zeros((1, resolution * ssaa, resolution * ssaa, 4), device=self.device, dtype=torch.float32)
                        individual_results.append(dummy_rast)
                
                if individual_results:
                    rast_batch = torch.cat(individual_results, dim=0)
                else:
                    # Complete fallback: return empty results
                    print("⚠️  All items failed, returning empty results")
                    default_img = torch.zeros((batch_size, resolution, resolution, 3), dtype=torch.float32, device=self.device)
                    ret_dict = {k: default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
                    return ret_dict
            else:
                raise e  # Re-raise non-CUDA-700 errors

        if return_rast_vertices:
            return rast_batch, full_proj_batch

        out_dict = edict()

        for type in return_types:
            img = None
            if type == "mask":
                img = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera_batch[..., 2:3].contiguous(), rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int)
            elif type == "normal":
                # Fix: Handle face_normal shape properly
                if hasattr(mesh, 'face_normal') and mesh.face_normal is not None:
                    if len(mesh.face_normal.shape) == 3 and mesh.face_normal.shape[1] == 3:
                        # If face_normal is [N, 3, 3], take the first normal vector
                        face_normals = mesh.face_normal[:, 0, :].reshape(1, -1, 3)
                    else:
                        # If face_normal is [N, 3], use directly
                        face_normals = mesh.face_normal.reshape(1, -1, 3)
                    
                    img = dr.interpolate(
                        face_normals, rast_batch,
                        torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                    )[0]
                    img = (img + 1) / 2
                    img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
                else:
                    # Fallback: use vertex normals from vertex_attrs
                    if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[1] >= 6:
                        vertex_normals = mesh.vertex_attrs[:, 3:6].contiguous()
                        img = dr.interpolate(vertex_normals, rast_batch, faces_int)[0]
                        img = (img + 1) / 2
                        img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
                    else:
                        # Final fallback: default normal
                        img = torch.ones_like(rast_batch[..., :3]) * 0.5
            elif type == "normal_map":
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast_batch, faces_int)[0]
                img = (img + 1) / 2
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
            elif type == "color":
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
            elif type == "envmap":
                # Sample envmap at each vertex using the SH approximation
                vert_light = params['sh'].eval(mesh.vertex_attrs[:, 3:].contiguous()).contiguous()
                # Sample incoming radiance
                light = dr.interpolate(vert_light[None, ...], rast_batch, faces_int)[0]

                col = torch.cat((light / torch.pi, torch.ones((*light.shape[:-1],1), device='cuda')), dim=-1)
                img = dr.antialias(torch.where(rast_batch[..., -1:] != 0, col, params['bgs']), rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)[..., :-1]

                # vert_light = torch.ones_like(mesh.vertex_attrs[:, 3:])
                # col = dr.interpolate(vert_light[None, ...], rast_batch, faces_int)[0]
                # img = dr.antialias(col, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)                
            
            '''Manually calculate the rendering process to visualize the rendering process'''
            # from PIL import Image
            # from torchvision.transforms import ToPILImage
            # to_pil = ToPILImage()

            # timg = to_pil(img[0].permute(2, 0, 1).detach().cpu())
            # timg.save("test.png")

            # mask = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)
            # mimg = to_pil(mask[0].permute(2, 0, 1).detach().cpu())
            # mimg.save("mask.png")

            # img = dr.interpolate(mesh.vertices, rast_batch, faces_int)[0]
            # img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)

            # pt3 = img[0][mask[0].squeeze() > 0].reshape(-1, 3)

            # pt3 = torch.cat([pt3, torch.ones_like(pt3[..., :1])], dim=-1)
            # pt3 = torch.matmul(pt3, full_proj_batch[0].transpose(-1, -2))

            # converted_pcd = pt3[..., :2] / pt3[..., -1:]
            # converted_pcd = (converted_pcd + 1) * mimg.size[0] / 2
            # converted_pcd = converted_pcd.clamp(min=0, max=mimg.size[0])
            # # init a black image and then asign the color of the point cloud
            # cimg = torch.ones((mimg.size[0], mimg.size[0], 3))
            # converted_pcd = converted_pcd.reshape(-1, 2).cpu().long().numpy()

            # # use converted_pcd.long() to index the image and then assign the corresponding color from images[0] to the indexed position
            # cimg[converted_pcd[..., 1], converted_pcd[..., 0]] = torch.tensor(np.array(timg))[mask[0].cpu().squeeze() > 0].reshape(-1, 3).float() / 255

            # cimg = cimg.reshape(mimg.size[0], mimg.size[0], 3)

            # cimg = to_pil(cimg.permute(2, 0, 1).detach().cpu())
            # cimg.save("cimg.png")
            
            '''Return the 3D points in the world coordinate space of the rendered image'''
            if type == "points3Dpos":
                img = dr.interpolate(mesh.vertices, rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)

                mask = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)

                # Since the znear zfar and etc has been determined, resolution here is on the fine-grained level of sampling, not spatial resolution
                if ssaa > 1:
                    img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    mask = F.interpolate(mask.permute(0, 3, 1, 2), (resolution, resolution), mode='nearest').permute(0, 2, 3, 1)

                # 修复维度问题：确保img和mask都有相同的batch维度进行索引
                if img.dim() == 4 and mask.dim() == 4:
                    # [B, H, W, 3] and [B, H, W, 1]
                    batch_points = []
                    for b in range(img.shape[0]):
                        single_img = img[b]  # [H, W, 3]
                        single_mask = mask[b].squeeze(-1)  # [H, W]
                        if single_mask.sum() > 0:  # 确保有有效的点
                            pt3 = single_img[single_mask > 0]  # [N, 3]
                            batch_points.append(pt3)
                    # 合并所有batch的点
                    if batch_points:
                        pt3 = torch.cat(batch_points, dim=0)
                    else:
                        pt3 = torch.empty((0, 3), device=img.device, dtype=img.dtype)
                elif img.dim() == 3 and mask.dim() == 3:
                    # [H, W, 3] and [H, W, 1]
                    mask_2d = mask.squeeze(-1)  # [H, W]
                    if mask_2d.sum() > 0:
                        pt3 = img[mask_2d > 0]  # [N, 3]
                    else:
                        pt3 = torch.empty((0, 3), device=img.device, dtype=img.dtype)
                else:
                    # Fallback: 尝试原始逻辑但加上安全检查
                    try:
                        mask_flat = mask.squeeze()
                        if mask_flat.dim() > 2:
                            mask_flat = mask_flat.view(-1)
                        if img.dim() > 3:
                            img_flat = img.view(-1, img.shape[-1])
                        else:
                            img_flat = img.view(-1, 3)
                        pt3 = img_flat[mask_flat > 0]
                    except Exception as e:
                        print(f"⚠️  Error in points3Dpos fallback: {e}")
                        pt3 = torch.empty((0, 3), device=img.device, dtype=img.dtype)

                out_dict[type] = pt3
            else:
                if ssaa > 1:
                    img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                    # 保持批次维度，不要squeeze掉batch dimension
                    # img 现在是 [B, C, H, W] 格式，这是我们想要的
                else:
                    img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    # 保持批次维度，不要squeeze掉batch dimension
                    # img 现在是 [B, C, H, W] 格式，这是我们想要的
                out_dict[type] = img

        return out_dict
