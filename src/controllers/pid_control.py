"""
PID控制器示例
"""

import numpy as np
from src.controllers.base_controller import BaseController
from typing import Dict, Any, Optional

class PIDController(BaseController):
    """PID控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # PID参数
        self.kp = config.get('kp', 2.0)
        self.ki = config.get('ki', 0.5)
        self.kd = config.get('kd', 1.0)
        
        # 积分项和前一次误差
        self.integral = 0.0
        self.prev_error = 0.0
        
    def control(self, state: np.ndarray, 
                reference: Optional[np.ndarray] = None,
                time: float = 0.0) -> float:
        """计算控制输入"""
        # 控制目标：让x2跟踪参考信号
        x2 = state[1]
        v2 = state[3]
        
        if reference is not None:
            x2_ref = reference[1] if len(reference) > 1 else reference[0]
        else:
            x2_ref = 0.0  # 默认设定点
        
        # 计算误差
        error = x2_ref - x2
        
        # 计算PID控制
        self.integral += error * 0.01  # 假设dt=0.01
        derivative = (error - self.prev_error) / 0.01 if self.prev_error != 0 else 0
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 更新前一次误差
        self.prev_error = error
        
        return control
    
    def reset(self) -> None:
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0
