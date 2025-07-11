import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import os

class CameraEstimationVisualizer:
    """相机估计可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.font_size = 12
        
    def create_frame_comparison_grid(self, frame_data: Dict, output_path: str) -> str:
        """
        创建单帧完整对比网格图
        
        Args:
            frame_data: 包含各阶段图像路径的字典
            output_path: 输出图像路径
            
        Returns:
            str: 保存的图像路径
        """
        # 2x3 网格布局
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(f"Frame {frame_data.get('frame_id', 'Unknown')} - Camera Estimation Progress", 
                    fontsize=16, fontweight='bold')
        
        # 定义布局和标题
        layout = [
            (0, 0, 'original', 'Original Image'),
            (0, 1, 'cropped', 'Cropped & RMBG'),  # 优先显示裁剪后的图像
            (0, 2, 'large_sampling', 'Large Sampling'),
            (1, 0, 'dust3r', 'Dust3R Init'),
            (1, 1, 'pso', 'PSO Result'),
            (1, 2, 'final_align', 'Final Alignment')
        ]
        
        for row, col, key, title in layout:
            ax = axes[row, col]
            
            # 如果 cropped 不存在，fallback 到 rmbg
            if key == 'cropped' and (key not in frame_data or not os.path.exists(frame_data.get(key, ''))):
                key = 'rmbg'
            
            if key in frame_data and frame_data[key] is not None:
                if isinstance(frame_data[key], str) and os.path.exists(frame_data[key]):
                    # 从文件加载图像
                    img = Image.open(frame_data[key])
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    ax.imshow(np.array(img))
                elif isinstance(frame_data[key], np.ndarray):
                    # 直接显示numpy数组
                    ax.imshow(frame_data[key])
                else:
                    # 显示占位图
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_facecolor('lightgray')
            else:
                # 显示占位图
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_facecolor('lightgray')
            
            ax.set_title(title, fontsize=self.font_size, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_optimization_progress(self, frame_data: Dict, output_path: str) -> str:
        """
        创建优化过程可视化（横向排列）
        
        Args:
            frame_data: 包含各阶段图像路径的字典
            output_path: 输出图像路径
            
        Returns:
            str: 保存的图像路径
        """
        # 1x4 横向布局显示优化过程
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Frame {frame_data.get('frame_id', 'Unknown')} - Optimization Progress", 
                    fontsize=16, fontweight='bold')
        
        # 定义优化过程的步骤
        steps = [
            ('large_sampling', 'Large Sampling'),
            ('dust3r', 'Dust3R Init'),
            ('pso', 'PSO Optimization'),
            ('final_align', 'Final Result')
        ]
        
        for i, (key, title) in enumerate(steps):
            ax = axes[i]
            
            if key in frame_data and frame_data[key] is not None:
                if isinstance(frame_data[key], str) and os.path.exists(frame_data[key]):
                    img = Image.open(frame_data[key])
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    ax.imshow(np.array(img))
                elif isinstance(frame_data[key], np.ndarray):
                    ax.imshow(frame_data[key])
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_facecolor('lightgray')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_facecolor('lightgray')
            
            ax.set_title(title, fontsize=self.font_size, fontweight='bold')
            ax.axis('off')
            
            # 添加箭头指示进度（除了最后一个）
            if i < len(steps) - 1:
                # 在子图之间添加箭头
                arrow = patches.FancyArrowPatch((0.95, 0.5), (1.05, 0.5), 
                                              connectionstyle="arc3,rad=0", 
                                              arrowstyle='->', 
                                              mutation_scale=20, 
                                              color='red',
                                              transform=ax.transAxes,
                                              clip_on=False)
                ax.add_patch(arrow)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_side_by_side_comparison(self, frame_data: Dict, output_path: str) -> str:
        """
        创建原图与最终结果的对比图
        
        Args:
            frame_data: 包含各阶段图像路径的字典
            output_path: 输出图像路径
            
        Returns:
            str: 保存的图像路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Frame {frame_data.get('frame_id', 'Unknown')} - Before vs After", 
                    fontsize=16, fontweight='bold')
        
        # 原图
        if 'original' in frame_data and frame_data['original'] is not None:
            if isinstance(frame_data['original'], str) and os.path.exists(frame_data['original']):
                img = Image.open(frame_data['original'])
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                axes[0].imshow(np.array(img))
            else:
                axes[0].text(0.5, 0.5, 'No Original', ha='center', va='center', 
                           transform=axes[0].transAxes, fontsize=12)
                axes[0].set_facecolor('lightgray')
        
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 最终结果
        if 'final_align' in frame_data and frame_data['final_align'] is not None:
            if isinstance(frame_data['final_align'], str) and os.path.exists(frame_data['final_align']):
                img = Image.open(frame_data['final_align'])
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                axes[1].imshow(np.array(img))
            else:
                axes[1].text(0.5, 0.5, 'No Result', ha='center', va='center', 
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].set_facecolor('lightgray')
        
        axes[1].set_title('Final Alignment Result', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def load_image_safely(self, image_path: str) -> Optional[np.ndarray]:
        """
        安全地加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Optional[np.ndarray]: 加载的图像数组，如果失败则返回None
        """
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                return np.array(img)
            return None
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None 