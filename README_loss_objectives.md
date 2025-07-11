# Loss Objective System for Camera Estimation

这个文档介绍了新的 loss objective 系统，允许你轻松测试和比较不同的损失函数。

## 系统概览

### 文件结构
```
📁 项目根目录/
├── utils/loss_objective.py          # Loss objective 定义
├── camera_estimation.py             # 主程序（已修改）
├── test_loss_objectives.py          # 批量测试脚本
└── README_loss_objectives.md        # 本文档
```

### 可用的 Loss Objectives

1. **dreamsim** - DreamSim 感知损失 + 边界一致性（原始实现）
2. **lpips** - LPIPS 感知损失 + 边界一致性
3. **l1l2** - L1 + L2 损失 + 边界一致性
4. **ssim** - SSIM 结构相似性损失 + 边界一致性
5. **hybrid** - 混合损失（DreamSim + LPIPS + L1）
6. **mask_only** - 仅使用 mask 损失（用于几何对齐测试）

## 使用方法

### 1. 查看所有可用的 Loss Objectives

```bash
python camera_estimation.py --list_losses
```

### 2. 使用特定的 Loss Objective

```bash
python camera_estimation.py \
    --input_dir /path/to/glb/files \
    --output_dir ./output_dreamsim \
    --source_images_dir /path/to/source/images \
    --model Hunyuan \
    --loss_type dreamsim
```

### 3. 测试不同的 Loss Objectives

#### 测试单个 Loss Objective:
```bash
python camera_estimation.py \
    --input_dir ./tmp \
    --output_dir ./results_lpips \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type lpips
```

#### 批量测试所有 Loss Objectives:
```bash
python test_loss_objectives.py \
    --input_dir ./tmp \
    --output_base_dir ./experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan
```

#### 测试特定的几个 Loss Objectives:
```bash
python test_loss_objectives.py \
    --input_dir ./tmp \
    --output_base_dir ./experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_types dreamsim lpips hybrid
```

### 4. 结果比较

批量测试完成后，会自动生成：
- `experiment_summary.txt` - 详细的实验结果报告
- `compare_results.py` - 可视化比较脚本

运行可视化比较：
```bash
cd ./experiments
python compare_results.py
```

## 实际使用示例

### 基于你当前的设置运行：

```bash
# 1. 测试 DreamSim（原始）
python camera_estimation.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_dir ./output_dreamsim \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type dreamsim

# 2. 测试 LPIPS
python camera_estimation.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_dir ./output_lpips \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type lpips

# 3. 批量测试所有 Loss Objectives
python test_loss_objectives.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_base_dir ./loss_comparison_experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan
```

## Loss Objective 详细说明

### 1. DreamSim Boundary Loss (dreamsim)
- **特点**: 基于深度学习的感知相似性，更符合人类视觉
- **组成**: DreamSim 损失 + 边界一致性损失
- **适用**: 一般情况下的最佳选择

### 2. Perceptual Loss (lpips)
- **特点**: VGG 特征感知损失
- **组成**: LPIPS 损失 + 边界一致性损失
- **适用**: 强调纹理和细节匹配

### 3. L1L2 Loss (l1l2)
- **特点**: 传统像素级损失
- **组成**: L1 损失 (0.5) + L2 损失 (0.5) + 边界一致性损失
- **适用**: 简单快速，适合调试

### 4. SSIM Loss (ssim)
- **特点**: 结构相似性损失
- **组成**: SSIM 损失 + 边界一致性损失
- **适用**: 强调结构和亮度一致性

### 5. Hybrid Loss (hybrid)
- **特点**: 多种损失的组合
- **组成**: DreamSim (0.4) + LPIPS (0.3) + L1 (0.2) + 边界一致性
- **适用**: 最全面但计算量大

### 6. Mask Only Loss (mask_only)
- **特点**: 仅几何对齐
- **组成**: 只有 mask 一致性损失
- **适用**: 测试几何对齐，忽略纹理

## 性能建议

### 速度排序（快到慢）:
1. `mask_only` - 最快
2. `l1l2` - 很快
3. `ssim` - 快
4. `dreamsim` - 中等
5. `lpips` - 慢
6. `hybrid` - 最慢

### 质量排序（经验估计）:
1. `hybrid` - 最好但最慢
2. `dreamsim` - 平衡最佳
3. `lpips` - 纹理细节好
4. `ssim` - 结构好
5. `l1l2` - 基础质量
6. `mask_only` - 仅几何

## 故障排除

### 常见问题:

1. **CUDA 内存不足**:
   ```bash
   # 使用较小的 batch size 或更简单的 loss
   --loss_type l1l2  # 或 mask_only
   ```

2. **某个 Loss Objective 失败**:
   ```bash
   # 检查依赖是否安装
   pip install lpips dreamsim
   ```

3. **结果差异很大**:
   - 不同的 loss objective 会产生不同的结果
   - 建议先用 `dreamsim`（原始）作为基准

## 扩展 Loss Objectives

如果你想添加新的 loss objective，在 `utils/loss_objective.py` 中：

1. 继承 `LossObjective` 类
2. 实现 `setup_models()` 和 `compute_loss()` 方法
3. 在 `create_loss_objective()` 函数中注册新的 loss type

示例:
```python
class CustomLoss(LossObjective):
    def setup_models(self):
        # 初始化你的模型
        pass
    
    def compute_loss(self, rendered_image, target_image, rendered_mask, target_mask, **kwargs):
        # 计算你的损失
        return loss_value
```

## 总结

这个 loss objective 系统让你能够：
- ✅ 轻松切换不同的损失函数
- ✅ 批量测试和比较结果
- ✅ 扩展新的损失函数
- ✅ 获得详细的实验报告和可视化比较

享受实验不同的 loss objectives！🎯 

这个文档介绍了新的 loss objective 系统，允许你轻松测试和比较不同的损失函数。

## 系统概览

### 文件结构
```
📁 项目根目录/
├── utils/loss_objective.py          # Loss objective 定义
├── camera_estimation.py             # 主程序（已修改）
├── test_loss_objectives.py          # 批量测试脚本
└── README_loss_objectives.md        # 本文档
```

### 可用的 Loss Objectives

1. **dreamsim** - DreamSim 感知损失 + 边界一致性（原始实现）
2. **lpips** - LPIPS 感知损失 + 边界一致性
3. **l1l2** - L1 + L2 损失 + 边界一致性
4. **ssim** - SSIM 结构相似性损失 + 边界一致性
5. **hybrid** - 混合损失（DreamSim + LPIPS + L1）
6. **mask_only** - 仅使用 mask 损失（用于几何对齐测试）

## 使用方法

### 1. 查看所有可用的 Loss Objectives

```bash
python camera_estimation.py --list_losses
```

### 2. 使用特定的 Loss Objective

```bash
python camera_estimation.py \
    --input_dir /path/to/glb/files \
    --output_dir ./output_dreamsim \
    --source_images_dir /path/to/source/images \
    --model Hunyuan \
    --loss_type dreamsim
```

### 3. 测试不同的 Loss Objectives

#### 测试单个 Loss Objective:
```bash
python camera_estimation.py \
    --input_dir ./tmp \
    --output_dir ./results_lpips \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type lpips
```

#### 批量测试所有 Loss Objectives:
```bash
python test_loss_objectives.py \
    --input_dir ./tmp \
    --output_base_dir ./experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan
```

#### 测试特定的几个 Loss Objectives:
```bash
python test_loss_objectives.py \
    --input_dir ./tmp \
    --output_base_dir ./experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_types dreamsim lpips hybrid
```

### 4. 结果比较

批量测试完成后，会自动生成：
- `experiment_summary.txt` - 详细的实验结果报告
- `compare_results.py` - 可视化比较脚本

运行可视化比较：
```bash
cd ./experiments
python compare_results.py
```

## 实际使用示例

### 基于你当前的设置运行：

```bash
# 1. 测试 DreamSim（原始）
python camera_estimation.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_dir ./output_dreamsim \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type dreamsim

# 2. 测试 LPIPS
python camera_estimation.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_dir ./output_lpips \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan \
    --loss_type lpips

# 3. 批量测试所有 Loss Objectives
python test_loss_objectives.py \
    --input_dir /home/zhiyuan_ma/code2/V2M4/results_examples_w_tracking/tmp \
    --output_base_dir ./loss_comparison_experiments \
    --source_images_dir ./examples3/tmp \
    --model Hunyuan
```

## Loss Objective 详细说明

### 1. DreamSim Boundary Loss (dreamsim)
- **特点**: 基于深度学习的感知相似性，更符合人类视觉
- **组成**: DreamSim 损失 + 边界一致性损失
- **适用**: 一般情况下的最佳选择

### 2. Perceptual Loss (lpips)
- **特点**: VGG 特征感知损失
- **组成**: LPIPS 损失 + 边界一致性损失
- **适用**: 强调纹理和细节匹配

### 3. L1L2 Loss (l1l2)
- **特点**: 传统像素级损失
- **组成**: L1 损失 (0.5) + L2 损失 (0.5) + 边界一致性损失
- **适用**: 简单快速，适合调试

### 4. SSIM Loss (ssim)
- **特点**: 结构相似性损失
- **组成**: SSIM 损失 + 边界一致性损失
- **适用**: 强调结构和亮度一致性

### 5. Hybrid Loss (hybrid)
- **特点**: 多种损失的组合
- **组成**: DreamSim (0.4) + LPIPS (0.3) + L1 (0.2) + 边界一致性
- **适用**: 最全面但计算量大

### 6. Mask Only Loss (mask_only)
- **特点**: 仅几何对齐
- **组成**: 只有 mask 一致性损失
- **适用**: 测试几何对齐，忽略纹理

## 性能建议

### 速度排序（快到慢）:
1. `mask_only` - 最快
2. `l1l2` - 很快
3. `ssim` - 快
4. `dreamsim` - 中等
5. `lpips` - 慢
6. `hybrid` - 最慢

### 质量排序（经验估计）:
1. `hybrid` - 最好但最慢
2. `dreamsim` - 平衡最佳
3. `lpips` - 纹理细节好
4. `ssim` - 结构好
5. `l1l2` - 基础质量
6. `mask_only` - 仅几何

## 故障排除

### 常见问题:

1. **CUDA 内存不足**:
   ```bash
   # 使用较小的 batch size 或更简单的 loss
   --loss_type l1l2  # 或 mask_only
   ```

2. **某个 Loss Objective 失败**:
   ```bash
   # 检查依赖是否安装
   pip install lpips dreamsim
   ```

3. **结果差异很大**:
   - 不同的 loss objective 会产生不同的结果
   - 建议先用 `dreamsim`（原始）作为基准

## 扩展 Loss Objectives

如果你想添加新的 loss objective，在 `utils/loss_objective.py` 中：

1. 继承 `LossObjective` 类
2. 实现 `setup_models()` 和 `compute_loss()` 方法
3. 在 `create_loss_objective()` 函数中注册新的 loss type

示例:
```python
class CustomLoss(LossObjective):
    def setup_models(self):
        # 初始化你的模型
        pass
    
    def compute_loss(self, rendered_image, target_image, rendered_mask, target_mask, **kwargs):
        # 计算你的损失
        return loss_value
```

## 总结

这个 loss objective 系统让你能够：
- ✅ 轻松切换不同的损失函数
- ✅ 批量测试和比较结果
- ✅ 扩展新的损失函数
- ✅ 获得详细的实验报告和可视化比较

享受实验不同的 loss objectives！🎯 
 