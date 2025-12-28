"""
仿真引擎 - 运行仿真循环
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
import yaml
from tqdm import tqdm  # 【新增】导入进度条库

class Simulator:
    """
    双质量弹簧系统仿真引擎
    """
    
    def __init__(self, plant, config_path: str):
        """
        初始化仿真器
        
        Args:
            plant: 植物模型实例
            config_path: 配置文件路径
        """
        self.plant = plant
        self.config = self._load_config(config_path)
        self.controller = None
        
        # 仿真参数
        self.dt = self.config['simulation']['dt']
        self.total_time = self.config['simulation']['total_time']
        self.integration_method = self.config['simulation']['integration_method']
        
        # 扰动参数
        self.disturbance_enabled = self.config['disturbance']['enabled']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        # 增加 encoding='utf-8' 以支持中文
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    
    def set_controller(self, controller) -> None:
        """设置控制器"""
        self.controller = controller
        
    def _get_disturbance(self, time: float) -> tuple:
        """计算扰动（支持正弦、阶跃、冲激噪声）"""
        if not self.disturbance_enabled:
            return (0.0, 0.0)
        
        w1, w2 = 0.0, 0.0
        
        # 正弦扰动
        if self.config['disturbance']['sin_disturbance']['enabled']:
            w1 = self.config['disturbance']['sin_disturbance']['w1_amplitude'] * np.sin(2 * np.pi * self.config['disturbance']['sin_disturbance']['frequency'] * time)
            w2 = self.config['disturbance']['sin_disturbance']['w2_amplitude'] * np.sin(2 * np.pi * self.config['disturbance']['sin_disturbance']['frequency'] * time)
        
        # 阶跃扰动
        if self.config['disturbance']['step_disturbance']['enabled']:
            step_config = self.config['disturbance']['step_disturbance']
            if time >= step_config['start_time']:
                magnitude = step_config['magnitude']
                direction = 1 if step_config['direction'] == "positive" else -1
                target = step_config['target']
                
                if target == "m1":
                    w1 = magnitude * direction
                elif target == "m2":
                    w2 = magnitude * direction
        
        # 冲激扰动
        if self.config['disturbance']['impulse_disturbance']['enabled']:
            impulse_config = self.config['disturbance']['impulse_disturbance']
            start_time = impulse_config['start_time']
            duration = impulse_config['duration']
            magnitude = impulse_config['magnitude']
            direction = 1 if impulse_config['direction'] == "positive" else -1
            target = impulse_config['target']
            
            if abs(time - start_time) < duration / 2:
                pulse_magnitude = magnitude / duration
                if target == "m1":
                    w1 = pulse_magnitude * direction
                elif target == "m2":
                    w2 = pulse_magnitude * direction
        
        return (w1, w2)


    
    def run(self, reference_generator: Optional[Callable] = None) -> Dict[str, np.ndarray]:
        """
        运行仿真
        
        Args:
            reference_generator: 参考轨迹生成器函数
            
        Returns:
            仿真结果字典
        """
        # 时间数组
        num_steps = int(self.total_time / self.dt) + 1
        time_array = np.linspace(0, self.total_time, num_steps)
        
        # 重置植物
        self.plant.reset()
        
        # 【新增】使用 tqdm 包裹循环，显示进度条
        # desc: 进度条前缀描述
        # unit: 进度单位（步）
        # ncols: 进度条宽度（设置为80使其适应更多终端）
        for i, t in enumerate(tqdm(time_array, desc="仿真进度", unit="步", ncols=80)):
            # 获取当前状态
            state = self.plant.get_state()
            
            # 计算参考信号
            reference = None
            if reference_generator is not None:
                reference = reference_generator(t)
            
            # 计算控制输入
            if self.controller is not None:
                control = self.controller.update(state, reference, t)
            else:
                control = 0.0
            
            # 获取扰动
            disturbances = self._get_disturbance(t)
            
            # 执行一步仿真
            self.plant.step(control, self.dt, self.integration_method, disturbances)
        
        # 返回仿真结果
        return self._get_results()
    
    def _get_results(self) -> Dict[str, np.ndarray]:
        """获取仿真结果"""
        history = np.array(self.plant.state_history)
        return {
            'time': np.array(self.plant.time_history),
            'x1': history[:, 0],
            'x2': history[:, 1],
            'v1': history[:, 2],
            'v2': history[:, 3],
            'control': np.array(self.plant.control_history),
            'disturbances': np.array(self.plant.disturbance_history),
            'vibration': history[:, 0] - history[:, 1]
        }
