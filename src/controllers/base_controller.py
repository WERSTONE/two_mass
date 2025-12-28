"""
控制器基类，用于实现各种控制算法
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseController(ABC):
    """
    控制器基类
    
    所有具体的控制器都应该继承此类并实现 control 方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化控制器
        
        Args:
            config: 控制器配置字典
        """
        self.config = config
        self.name = self.__class__.__name__
        self.reset()
        
    @abstractmethod
    def control(self, state: np.ndarray, 
                reference: Optional[np.ndarray] = None,
                time: float = 0.0) -> float:
        """
        计算控制输入
        
        Args:
            state: 当前状态向量 [x1, x2, v1, v2]
            reference: 参考状态/轨迹
            time: 当前时间
            
        Returns:
            控制输入 u
        """
        pass
    
    def reset(self) -> None:
        """重置控制器内部状态"""
        pass
    
    def update(self, state: np.ndarray, 
              reference: Optional[np.ndarray] = None,
              time: float = 0.0) -> float:
        """
        更新控制器并计算控制输入
        
        Args:
            state: 当前状态向量
            reference: 参考状态
            time: 当前时间
            
        Returns:
            控制输入 u
        """
        return self.control(state, reference, time)
    
    def get_name(self) -> str:
        """获取控制器名称"""
        return self.name
