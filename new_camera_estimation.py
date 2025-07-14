#!/usr/bin/env python3
"""
新的相机估计入口文件 - 完全兼容原始API
基于SimpleCamEstimate架构，模块化设计，性能优化
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# 导入现有模块
from v2m4_trellis.utils import render_utils
from v2m4_trellis.renderers import MeshRenderer
from modules.loss_objective import create_loss_objective
from modules.new_camera_estimator import NewCameraEstimator


def load_glb_to_mesh_extract_result(glb_path):
    """
    加载GLB文件 - 保持与原始API完全一致
    """
    # 直接使用原始实现
    from camera_estimation import load_glb_to_mesh_extract_result as original_load
    return original_load(glb_path)


def find_closet_camera_pos_with_custom_loss(sample, rmbg_image, loss_objective, 
                                           resolution=512, bg_color=(0, 0, 0), 
                                           iterations=100, params=None, 
                                           return_optimize=False, prior_params=None, 
                                           save_path=None, is_Hunyuan=False, 
                                           use_vggt=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    新的相机姿态估计函数 - 完全兼容原始API
    
    使用基于SimpleCamEstimate的新架构，但保持接口不变
    
    Args:
        sample: 3D mesh sample
        rmbg_image: 去背景后的图像
        loss_objective: 损失函数对象
        resolution: 渲染分辨率
        bg_color: 背景颜色
        iterations: 迭代次数
        params: 预设参数
        return_optimize: 是否返回优化结果
        prior_params: 先验参数
        save_path: 保存路径
        is_Hunyuan: 是否是Hunyuan模型
        use_vggt: 是否使用VGGT
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (渲染图像, 相机参数)
    """
    
    # 设置渲染器
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 2
    
    # 创建新的估计器
    estimator = NewCameraEstimator(loss_objective, renderer)
    
    # 调用估计函数
    return estimator.find_camera_pose(
        sample, rmbg_image, resolution, bg_color, iterations, params,
        return_optimize, prior_params, save_path, is_Hunyuan, use_vggt
    )


def parse_args():
    """解析命令行参数 - 与原始API保持一致"""
    # 直接使用原始实现
    from camera_estimation import parse_args as original_parse_args
    return original_parse_args()


def main():
    """主函数 - 保持与原始API一致"""
    args = parse_args()
    
    # 导入原始模块
    import camera_estimation
    
    # 替换关键函数为新的实现
    camera_estimation.find_closet_camera_pos_with_custom_loss = find_closet_camera_pos_with_custom_loss
    
    # 运行原始主函数
    camera_estimation.main()


if __name__ == "__main__":
    main() 