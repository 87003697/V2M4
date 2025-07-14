"""
相机姿态数据结构 - 从SimpleCamEstimate复制并适配V2M4
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from v2m4_trellis.utils import render_utils


@dataclass
class CameraPose:
    """相机姿态参数 - 使用球坐标系，兼容V2M4系统"""
    elevation: float    # 仰角 (度)，正值表示相机在物体上方  
    azimuth: float      # 方位角 (度)，绕垂直轴旋转
    radius: float       # 相机到目标点的距离
    center_x: float = 0.0    # 目标点x坐标
    center_y: float = 0.0    # 目标点y坐标
    center_z: float = 0.0    # 目标点z坐标
    
    @property
    def target_point(self) -> Tuple[float, float, float]:
        """获取目标点坐标"""
        return (self.center_x, self.center_y, self.center_z)
    
    def to_yaw_pitch_params(self) -> Tuple[float, float, float, float, float, float]:
        """转换为V2M4原始的yaw/pitch参数格式"""
        yaw = np.radians(self.azimuth)
        pitch = np.radians(self.elevation)
        r = self.radius
        lookat_x = self.center_x
        lookat_y = self.center_y
        lookat_z = self.center_z
        
        return (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
    
    @classmethod
    def from_yaw_pitch_params(cls, yaw, pitch, r, lookat_x, lookat_y, lookat_z):
        """从V2M4原始参数创建CameraPose"""
        return cls(
            elevation=np.degrees(pitch),
            azimuth=np.degrees(yaw),
            radius=r,
            center_x=lookat_x,
            center_y=lookat_y,
            center_z=lookat_z
        )
    
    def __str__(self) -> str:
        return f"CameraPose(elev={self.elevation:.1f}°, azim={self.azimuth:.1f}°, r={self.radius:.2f})"


class CameraPoseConverter:
    """相机姿态转换工具"""
    
    @staticmethod
    def to_v2m4_format(pose: CameraPose) -> np.ndarray:
        """转换为V2M4格式的numpy数组"""
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = pose.to_yaw_pitch_params()
        return np.array([yaw, pitch, r, lookat_x, lookat_y, lookat_z])
    
    @staticmethod
    def from_v2m4_format(params: np.ndarray) -> CameraPose:
        """从V2M4格式的numpy数组创建CameraPose"""
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        return CameraPose.from_yaw_pitch_params(yaw, pitch, r, lookat_x, lookat_y, lookat_z) 