"""
PSO优化器模块 - 从SimpleCamEstimate复制并适配V2M4
"""

import torch
import random
import numpy as np
from typing import List, Callable, Tuple, Dict, Any
from .camera_pose import CameraPose


class PSOOptimizer:
    """粒子群优化器 - 适配V2M4的渲染系统"""
    
    def __init__(self, num_particles: int = 30, max_iterations: int = 50, 
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        
        # 默认搜索边界 - 与V2M4原始实现兼容
        self.default_bounds = {
            'elevation': (-45, 75),
            'azimuth': (-180, 180),
            'radius': (0.5, 8.0),
            'center_x': (-0.5, 0.5),
            'center_y': (-0.5, 0.5),
            'center_z': (-0.5, 0.5)
        }
    
    def optimize_batch(self, batch_objective_func: Callable[[List[CameraPose]], List[float]], 
                      candidates: List[CameraPose], 
                      bounds: Dict[str, Tuple[float, float]] = None) -> CameraPose:
        """PSO优化 - 批量渲染版本（性能优化）"""
        
        if bounds is None:
            bounds = self.default_bounds
        
        print(f"   🚀 Using batch PSO optimization ({self.num_particles} particles, {self.max_iterations} iterations)")
        
        # 从候选姿态中选择初始种群
        if len(candidates) >= self.num_particles:
            particles = candidates[:self.num_particles]
        else:
            particles = candidates + self._generate_random_particles(
                self.num_particles - len(candidates), bounds)
        
        # 初始化速度和个体最优
        velocities = [self._initialize_velocity() for _ in particles]
        personal_best = particles.copy()
        personal_best_scores = batch_objective_func(personal_best)
        
        # 全局最优
        global_best_idx = torch.argmin(torch.tensor(personal_best_scores)).item()
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        print(f"   📊 Initial best score: {global_best_score:.4f}")
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 更新所有粒子
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = torch.rand(2).tolist()
                
                # 计算速度更新
                inertia = self._multiply_velocity(velocities[i], self.w)
                cognitive = self._multiply_velocity(
                    self._subtract_poses(personal_best[i], particles[i]), 
                    self.c1 * r1)
                social = self._multiply_velocity(
                    self._subtract_poses(global_best, particles[i]), 
                    self.c2 * r2)
                
                velocities[i] = self._add_velocities([inertia, cognitive, social])
                
                # 更新位置
                particles[i] = self._add_pose_velocity(particles[i], velocities[i])
                particles[i] = self._clamp_pose(particles[i], bounds)
            
            # 批量评估所有粒子
            scores = batch_objective_func(particles)
            
            # 更新个体最优
            for i in range(self.num_particles):
                if scores[i] < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = scores[i]
                    
                    # 更新全局最优
                    if scores[i] < global_best_score:
                        global_best = particles[i]
                        global_best_score = scores[i]
            
            # 进度报告
            if (iteration + 1) % 10 == 0:
                print(f"   📈 Iteration {iteration + 1}/{self.max_iterations}: best score = {global_best_score:.4f}")
        
        print(f"   ✅ Batch PSO completed. Final best score: {global_best_score:.4f}")
        return global_best
    
    def _generate_random_particles(self, num: int, bounds: Dict[str, Tuple[float, float]]) -> List[CameraPose]:
        """生成随机粒子"""
        particles = []
        for _ in range(num):
            pose = CameraPose(
                elevation=random.uniform(*bounds['elevation']),
                azimuth=random.uniform(*bounds['azimuth']),
                radius=random.uniform(*bounds['radius']),
                center_x=random.uniform(*bounds['center_x']),
                center_y=random.uniform(*bounds['center_y']),
                center_z=random.uniform(*bounds['center_z'])
            )
            particles.append(pose)
        return particles
    
    def _initialize_velocity(self) -> Dict[str, float]:
        """初始化速度"""
        return {
            'elevation': random.uniform(-10, 10),
            'azimuth': random.uniform(-10, 10),
            'radius': random.uniform(-0.5, 0.5),
            'center_x': random.uniform(-0.1, 0.1),
            'center_y': random.uniform(-0.1, 0.1),
            'center_z': random.uniform(-0.1, 0.1)
        }
    
    def _multiply_velocity(self, velocity: Dict[str, float], factor: float) -> Dict[str, float]:
        """速度乘以因子"""
        return {k: v * factor for k, v in velocity.items()}
    
    def _subtract_poses(self, pose1: CameraPose, pose2: CameraPose) -> Dict[str, float]:
        """姿态相减"""
        return {
            'elevation': pose1.elevation - pose2.elevation,
            'azimuth': pose1.azimuth - pose2.azimuth,
            'radius': pose1.radius - pose2.radius,
            'center_x': pose1.center_x - pose2.center_x,
            'center_y': pose1.center_y - pose2.center_y,
            'center_z': pose1.center_z - pose2.center_z
        }
    
    def _add_velocities(self, velocities: List[Dict[str, float]]) -> Dict[str, float]:
        """速度相加"""
        result = {}
        for key in velocities[0].keys():
            result[key] = sum(v[key] for v in velocities)
        return result
    
    def _add_pose_velocity(self, pose: CameraPose, velocity: Dict[str, float]) -> CameraPose:
        """姿态加速度"""
        return CameraPose(
            elevation=pose.elevation + velocity['elevation'],
            azimuth=pose.azimuth + velocity['azimuth'],
            radius=pose.radius + velocity['radius'],
            center_x=pose.center_x + velocity['center_x'],
            center_y=pose.center_y + velocity['center_y'],
            center_z=pose.center_z + velocity['center_z']
        )
    
    def _clamp_pose(self, pose: CameraPose, bounds: Dict[str, Tuple[float, float]]) -> CameraPose:
        """限制姿态在边界内"""
        return CameraPose(
            elevation=max(bounds['elevation'][0], min(bounds['elevation'][1], pose.elevation)),
            azimuth=max(bounds['azimuth'][0], min(bounds['azimuth'][1], pose.azimuth)),
            radius=max(bounds['radius'][0], min(bounds['radius'][1], pose.radius)),
            center_x=max(bounds['center_x'][0], min(bounds['center_x'][1], pose.center_x)),
            center_y=max(bounds['center_y'][0], min(bounds['center_y'][1], pose.center_y)),
            center_z=max(bounds['center_z'][0], min(bounds['center_z'][1], pose.center_z))
        ) 