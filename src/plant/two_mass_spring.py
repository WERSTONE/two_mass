"""
B.2.13 Two-Mass-Spring Plant (Generic Flexible Space Structure)
简化的柔性空间结构模型，由两个通过弹簧连接的质量组成
"""

import numpy as np
from typing import Tuple, Dict, Any

class TwoMassSpringPlant:
    """
    双质量弹簧系统植物模型
    
    系统动力学方程 (考虑自然长度 L0):
    X1 = V1, X2 = V2,
    m*V1 = k*((x2 - x1) - L0) + u + w1,
    m*V2 = -k*((x2 - x1) - L0) + w2
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化双质量弹簧系统"""
        # 保存配置，供 reset 使用
        self.config = config
        
        self.mass = config.get('physical', {}).get('mass', 1.0)
        self.k = config.get('physical', {}).get('spring_constant', 1.0)
        # 【新增】读取弹簧自然长度，默认为0（兼容旧模型）
        self.natural_length = config.get('physical', {}).get('natural_length', 0.0)
        
        # 调用缺失的方法获取初始状态
        self.state = self._get_initial_state(config)
        self.time = 0.0
        
        # 历史数据
        self.time_history = [self.time]
        self.state_history = [self.state.copy()]
        self.control_history = [0.0]
        self.disturbance_history = [[0.0, 0.0]]
        
    def _get_initial_state(self, config: Dict[str, Any]) -> np.ndarray:
        """根据配置获取初始状态向量 [x1, x2, v1, v2]"""
        initial = config.get('initial_conditions', {})
        x1 = initial.get('x1', 0.0)
        x2 = initial.get('x2', 0.0)
        v1 = initial.get('v1', 0.0)
        v2 = initial.get('v2', 0.0)
        return np.array([x1, x2, v1, v2])
    
    def get_state(self) -> np.ndarray:
        """获取当前状态向量 [x1, x2, v1, v2]"""
        return self.state.copy()
    
    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典"""
        return {
            'x1': self.state[0],  # 质量体1位移
            'x2': self.state[1],  # 质量体2位移
            'v1': self.state[2],  # 质量体1速度
            'v2': self.state[3],  # 质量体2速度
            'time': self.time
        }
    
    def step(self, control: float, dt: float, 
             integration_method: str = 'rk4',
             disturbances: Tuple[float, float] = (0.0, 0.0)) -> None:
        """执行一步仿真"""
        state = self.state.copy()
        
        if integration_method == 'rk4':
            k1 = self._get_derivative(state, control, disturbances)
            k2 = self._get_derivative(state + 0.5 * dt * k1, control, disturbances)
            k3 = self._get_derivative(state + 0.5 * dt * k2, control, disturbances)
            k4 = self._get_derivative(state + dt * k3, control, disturbances)
            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else: # 默认回退到欧拉法，虽然配置里是rk4，但为了鲁棒性
            k1 = self._get_derivative(state, control, disturbances)
            new_state = state + dt * k1
            
        self.state = new_state
        self.time += dt
        self.time_history.append(self.time)
        self.state_history.append(self.state.copy())
        self.control_history.append(control)
        self.disturbance_history.append(list(disturbances))
    
    def _get_derivative(self, state: np.ndarray, control: float, 
                      disturbances: Tuple[float, float]) -> np.ndarray:
        """计算状态导数"""
        x1, x2, v1, v2 = state
        w1, w2 = disturbances
        # 【修改】弹簧力计算包含自然长度
        spring_force = self.k * ((x2 - x1) - self.natural_length)
        a1 = (spring_force + control + w1) / self.mass
        a2 = (-spring_force + w2) / self.mass
        return np.array([v1, v2, a1, a2])
    
    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取状态空间矩阵 (A, B, B_w)"""
        # 注意：这里的 A 矩阵依然是线性的，围绕平衡点的线性化结果与自然长度无关
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-self.k/self.mass, self.k/self.mass, 0, 0],
            [self.k/self.mass, -self.k/self.mass, 0, 0]
        ])
        
        B = np.array([[0], [0], [1/self.mass], [0]])
        B_w = np.array([[0, 0], [0, 0], [1/self.mass, 0], [0, 1/self.mass]])
        
        return A, B, B_w

    def reset(self) -> None:
        """重置植物状态到初始条件"""
        self.state = self._get_initial_state(self.config)
        self.time = 0.0
        self.time_history = [self.time]
        self.state_history = [self.state.copy()]
        self.control_history = [0.0]
        self.disturbance_history = [[0.0, 0.0]]

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统物理信息和特征值"""
        natural_freq = np.sqrt(self.k / self.mass)
        A, _, _ = self.get_matrices()
        eigenvalues = np.linalg.eigvals(A)
        return {
            'mass': self.mass,
            'spring_constant': self.k,
            'natural_length': self.natural_length, # 返回自然长度信息
            'natural_frequency': natural_freq,
            'eigenvalues': eigenvalues
        }
