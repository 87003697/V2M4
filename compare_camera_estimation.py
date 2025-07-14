#!/usr/bin/env python3
"""
相机估计系统对比测试脚本
比较原始系统和新系统的效果
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def load_test_image(image_path):
    """加载测试图像"""
    try:
        test_image = Image.open(image_path).convert('RGB')
        print(f"📸 Test image loaded from: {image_path}")
        print(f"📐 Image size: {test_image.size}")
        return test_image
    except Exception as e:
        print(f"❌ Error loading test image: {e}")
        return None

def create_test_image():
    """创建一个测试图像"""
    # 创建一个简单的测试图像
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # 添加一个圆形
    center = (256, 256)
    radius = 100
    y, x = np.ogrid[:512, :512]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = [255, 100, 100]  # 红色圆形
    
    # 添加一个矩形
    img[200:300, 350:450] = [100, 255, 100]  # 绿色矩形
    
    return Image.fromarray(img)

def test_camera_estimation_system(system_name, glb_path, test_image, loss_type="dreamsim"):
    """测试相机估计系统"""
    print(f"\n🧪 Testing {system_name} system...")
    
    try:
        if system_name == "original":
            from camera_estimation import find_closet_camera_pos_with_custom_loss, load_glb_to_mesh_extract_result
            from modules.loss_objective import create_loss_objective
        else:
            from new_camera_estimation import find_closet_camera_pos_with_custom_loss, load_glb_to_mesh_extract_result
            from modules.loss_objective import create_loss_objective
        
        # 加载GLB文件
        print(f"📁 Loading GLB file: {glb_path}")
        mesh_sample = load_glb_to_mesh_extract_result(glb_path)
        
        # 创建损失函数
        loss_objective = create_loss_objective(loss_type)
        
        # 运行相机估计
        start_time = time.time()
        
        save_path = f"test_output_{system_name}"
        os.makedirs(save_path, exist_ok=True)
        
        rendered_image, camera_params = find_closet_camera_pos_with_custom_loss(
            mesh_sample, 
            test_image, 
            loss_objective,
            resolution=512,
            iterations=50,  # 减少迭代次数以加快测试
            save_path=os.path.join(save_path, f"{system_name}_result")
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 保存结果
        result_path = os.path.join(save_path, f"{system_name}_rendered.png")
        imageio.imsave(result_path, rendered_image)
        
        print(f"✅ {system_name} system completed in {elapsed_time:.2f}s")
        print(f"📊 Camera parameters: {camera_params}")
        print(f"💾 Result saved to: {result_path}")
        
        return {
            'rendered_image': rendered_image,
            'camera_params': camera_params,
            'elapsed_time': elapsed_time,
            'result_path': result_path
        }
        
    except Exception as e:
        print(f"❌ Error in {system_name} system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(original_result, new_result, test_image):
    """比较两个系统的结果"""
    print("\n📊 Comparing results...")
    
    if original_result is None or new_result is None:
        print("❌ Cannot compare - one or both systems failed")
        return
    
    # 时间对比
    print(f"⏱️  Time comparison:")
    print(f"   Original: {original_result['elapsed_time']:.2f}s")
    print(f"   New:      {new_result['elapsed_time']:.2f}s")
    speedup = original_result['elapsed_time'] / new_result['elapsed_time']
    print(f"   Speedup:  {speedup:.2f}x")
    
    # 参数对比
    print(f"📐 Camera parameters comparison:")
    orig_params = original_result['camera_params']
    new_params = new_result['camera_params']
    
    param_names = ['yaw', 'pitch', 'radius', 'lookat_x', 'lookat_y', 'lookat_z']
    for i, name in enumerate(param_names):
        if i < len(orig_params) and i < len(new_params):
            diff = abs(orig_params[i] - new_params[i])
            print(f"   {name}: {orig_params[i]:.4f} vs {new_params[i]:.4f} (diff: {diff:.4f})")
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(np.array(test_image))
    axes[0].set_title('Target Image')
    axes[0].axis('off')
    
    # 原始系统结果
    axes[1].imshow(original_result['rendered_image'])
    axes[1].set_title(f'Original System\n({original_result["elapsed_time"]:.2f}s)')
    axes[1].axis('off')
    
    # 新系统结果
    axes[2].imshow(new_result['rendered_image'])
    axes[2].set_title(f'New System\n({new_result["elapsed_time"]:.2f}s)')
    axes[2].axis('off')
    
    plt.tight_layout()
    comparison_path = 'comparison_result.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Comparison saved to: {comparison_path}")
    
    # 计算图像相似度
    try:
        from skimage.metrics import structural_similarity as ssim
        import cv2
        
        # 转换为灰度
        orig_gray = cv2.cvtColor(original_result['rendered_image'], cv2.COLOR_RGB2GRAY)
        new_gray = cv2.cvtColor(new_result['rendered_image'], cv2.COLOR_RGB2GRAY)
        
        similarity = ssim(orig_gray, new_gray)
        print(f"🔍 Image similarity (SSIM): {similarity:.4f}")
        
        if similarity > 0.9:
            print("✅ Images are very similar - systems produce consistent results")
        elif similarity > 0.7:
            print("⚠️  Images are moderately similar - minor differences detected")
        else:
            print("❌ Images are quite different - significant differences detected")
            
    except ImportError:
        print("ℹ️  skimage not available - skipping similarity calculation")
    except Exception as e:
        print(f"⚠️  Error calculating similarity: {e}")

def main():
    print("🚀 Starting Camera Estimation System Comparison Test")
    print("=" * 60)
    
    # 准备测试数据 - 使用指定的文件
    image_path = "examples3/tmp/0000.png"
    glb_path = "results_examples/tmp/0000_re-canonicalization_sample.glb"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    if not os.path.exists(glb_path):
        print(f"❌ GLB file not found: {glb_path}")
        print("📁 Available GLB files:")
        os.system("find . -name '*.glb' | head -5")
        return
    
    # 加载测试图像
    test_image = load_test_image(image_path)
    if test_image is None:
        return
    
    # 测试原始系统
    original_result = test_camera_estimation_system("original", glb_path, test_image)
    
    # 测试新系统
    new_result = test_camera_estimation_system("new", glb_path, test_image)
    
    # 比较结果
    compare_results(original_result, new_result, test_image)
    
    print("\n🎉 Comparison test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 