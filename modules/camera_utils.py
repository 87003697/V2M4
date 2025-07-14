"""
相机估计工具函数 - 从SimpleCamEstimate复制并适配V2M4
"""

import torch
import cv2
import gc
import numpy as np
from PIL import Image
from typing import List, Optional
import torch.nn.functional as F
from .camera_pose import CameraPose


def preprocess_image(image: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """图像预处理"""
    # 确保图像是HWC格式
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # 如果是CHW格式，转换为HWC
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    
    # 调整大小
    if image.shape[:2] != (target_size, target_size):
        # 转换为numpy进行resize，因为cv2需要numpy
        image_np = image.cpu().numpy()
        image_np = cv2.resize(image_np, (target_size, target_size))
        image = torch.from_numpy(image_np).to(image.device)
    
    # 确保值在[0,1]范围内
    if image.max() > 1.0:
        image = image / 255.0
    
    return image


def batch_render_and_compare(mesh_sample, poses: List[CameraPose], target_image: torch.Tensor, 
                           target_mask: torch.Tensor, loss_objective, renderer, 
                           sub_batch_size: int = 8) -> List[float]:
    """批量渲染并计算损失"""
    device = target_image.device
    losses = []
    
    # 转换为V2M4格式的参数
    from .camera_pose import CameraPoseConverter
    params_list = [CameraPoseConverter.to_v2m4_format(pose) for pose in poses]
    
    # 导入V2M4的渲染函数
    from v2m4_trellis.utils.render_utils import (
        batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics
    )
    
    # 转换为张量
    params = torch.tensor(params_list, dtype=torch.float32).cuda()
    yaw, pitch, r, lookat_x, lookat_y, lookat_z = params.chunk(6, dim=1)
    lookat = torch.cat([lookat_x, lookat_y, lookat_z], dim=1)
    
    # 获取相机参数
    fov = 40
    extr, intr = batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaw, pitch, r, fov, lookat
    )
    
    # 分批渲染
    for i in range(0, len(poses), sub_batch_size):
        sub_extr = extr[i:i+sub_batch_size]
        sub_intr = intr[i:i+sub_batch_size]
        current_batch_size = sub_extr.shape[0]
        
        # 渲染
        res = renderer.render_batch(mesh_sample, sub_extr, sub_intr, return_types=["mask", "color"])
        rendering = torch.clip(res['color'], 0., 1.)  # [current_batch_size, 3, 512, 512]
        
        # 计算每个渲染结果的损失
        for j in range(current_batch_size):
            single_rendering = rendering[j]  # [3, 512, 512]
            single_mask = res['mask'][j]     # [1, 512, 512]
            
            # 确保mask格式正确
            if single_mask.dim() == 3 and single_mask.shape[0] == 1:
                single_mask = single_mask.squeeze(0)
            
            # 计算损失
            try:
                loss = loss_objective.compute_loss(
                    rendered_image=single_rendering,
                    target_image=target_image,
                    rendered_mask=single_mask,
                    target_mask=target_mask
                )
                losses.append(loss.item())
            except Exception as e:
                print(f"⚠️ Error computing loss: {e}")
                losses.append(float('inf'))
    
    return losses


def generate_sphere_samples(num_samples: int = 1000, elevation_range: tuple = (-45, 75), 
                          azimuth_range: tuple = (-180, 180), 
                          radius_range: tuple = (0.5, 8.0)) -> List[CameraPose]:
    """生成球面采样的相机姿态"""
    poses = []
    
    for _ in range(num_samples):
        # 随机采样
        elevation = np.random.uniform(*elevation_range)
        azimuth = np.random.uniform(*azimuth_range)
        radius = np.random.uniform(*radius_range)
        
        # 轻微变化中心点
        center_x = np.random.uniform(-0.1, 0.1)
        center_y = np.random.uniform(-0.1, 0.1)
        center_z = np.random.uniform(-0.1, 0.1)
        
        pose = CameraPose(
            elevation=elevation,
            azimuth=azimuth,
            radius=radius,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z
        )
        poses.append(pose)
    
    return poses


def generate_dust3r_candidates(reference_image: torch.Tensor, num_candidates: int = 10) -> List[CameraPose]:
    """生成基于DUSt3R的候选姿态（简化版本）"""
    # 这里简化实现，实际应该调用DUSt3R模型
    # 为了兼容，我们生成一些合理的候选姿态
    candidates = []
    
    # 生成一些常见的相机视角
    common_elevations = [0, 15, 30, 45, 60]
    common_azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    common_radii = [2.0, 3.0, 4.0, 5.0]
    
    for _ in range(num_candidates):
        elevation = np.random.choice(common_elevations) + np.random.uniform(-10, 10)
        azimuth = np.random.choice(common_azimuths) + np.random.uniform(-20, 20)
        radius = np.random.choice(common_radii) + np.random.uniform(-0.5, 0.5)
        
        # 轻微变化中心点
        center_x = np.random.uniform(-0.1, 0.1)
        center_y = np.random.uniform(-0.1, 0.1)
        center_z = np.random.uniform(-0.1, 0.1)
        
        pose = CameraPose(
            elevation=elevation,
            azimuth=azimuth,
            radius=radius,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z
        )
        candidates.append(pose)
    
    return candidates


def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def save_pose_visualization(pose: CameraPose, rendered_image: np.ndarray, 
                          target_image: np.ndarray, save_path: str):
    """保存姿态可视化结果"""
    try:
        import imageio
        import matplotlib.pyplot as plt
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 显示渲染结果
        axes[0].imshow(rendered_image)
        axes[0].set_title(f'Rendered\n{pose}')
        axes[0].axis('off')
        
        # 显示目标图像
        axes[1].imshow(target_image)
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save visualization: {e}")


def compute_pose_distance(pose1: CameraPose, pose2: CameraPose) -> float:
    """计算两个相机姿态之间的距离"""
    # 欧几里得距离
    elev_diff = (pose1.elevation - pose2.elevation) ** 2
    azim_diff = (pose1.azimuth - pose2.azimuth) ** 2
    radius_diff = (pose1.radius - pose2.radius) ** 2
    center_diff = ((pose1.center_x - pose2.center_x) ** 2 + 
                   (pose1.center_y - pose2.center_y) ** 2 + 
                   (pose1.center_z - pose2.center_z) ** 2)
    
    return np.sqrt(elev_diff + azim_diff + radius_diff + center_diff)


def filter_similar_poses(poses: List[CameraPose], threshold: float = 5.0) -> List[CameraPose]:
    """过滤相似的姿态"""
    if len(poses) <= 1:
        return poses
    
    filtered = [poses[0]]
    
    for pose in poses[1:]:
        # 检查是否与已有姿态太相似
        too_similar = False
        for existing_pose in filtered:
            if compute_pose_distance(pose, existing_pose) < threshold:
                too_similar = True
                break
        
        if not too_similar:
            filtered.append(pose)
    
    return filtered 