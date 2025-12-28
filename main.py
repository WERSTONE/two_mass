"""
主入口文件 - 双质量弹簧系统仿真
使用 MPC 控制器
"""

import sys
import os
import numpy as np
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.plant.two_mass_spring import TwoMassSpringPlant
from src.simulation.simulator import Simulator
from src.visualization.animator import Animator
# 导入新的 MPC 控制器
from src.controllers.mpc_control import MPCController
import mpc_plotting as plotter

def main():
    """主函数"""
    # 1. 加载配置
    config_path = 'config/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    # 2. 创建植物模型
    plant = TwoMassSpringPlant(full_config)
    
    # 3. 创建仿真器
    simulator = Simulator(plant, config_path)
    
    # --- 定义 MPC 控制器的参数 ---
    mpc_params = {
        'prediction_horizon': 300,  # 预测未来 300 步 (3秒)
        
        # 代价函数权重 Q (对应 x1, x2, v1, v2)
        'q_x1': 0.0,   # 不关心 m1 的绝对位置
        'q_x2': 100.0,  # 强力约束 m2 的位置到 0
        'q_v1': 0.0,   # 阻尼 m1 速度
        'q_v2': 1.0,   # 阻尼 m2 速度
        
        'r_input': 1.0,   # 惩制输入惩罚 (防止控制量过大)
        
        # 【新增】传递自然长度参数给控制器
        'natural_length': 1.0  
    }
    
    # --- 【新增】从配置中提取初始状态 ---
    init_cond = full_config.get('initial_conditions', {})
    initial_state_vector = np.array([
        init_cond.get('x1', 0.0),
        init_cond.get('x2', 0.0),
        init_cond.get('v1', 0.0),
        init_cond.get('v2', 0.0)
    ])
    
    # --- 【修改】实例化控制器对象时传入初始状态 ---
    mpc_controller = MPCController(mpc_params, initial_state=initial_state_vector)
    
    # 将控制器实例设置给仿真器
    simulator.set_controller(mpc_controller)

    def ref_generator(t: float) -> np.ndarray:
        """
        返回当前时刻的参考状态：[x1_ref, x2_ref, v1_ref, v2_ref]
        我们只关心 x2_ref，其他可以给 0
        """
        x2_ref = 1 # 保持初始位置

        return np.array([0.0, x2_ref, 0.0, 0.0])

    # 运行仿真
    print("运行 MPC 控制仿真...")
    results = simulator.run(reference_generator=ref_generator)
    print(f"x1的轨迹为{results['x1']}")
    print(f"x2的轨迹为{results['x2']}")
    
    # 可视化结果
    print("绘制结果...")
    animator = Animator(simulator.config)
    animator.animate_system(results)
    
    # 显示系统信息
    print("\n系统信息:")
    info = plant.get_system_info()
    plotter.generate_publication_plots(results, simulator.config, info, output_dir='./figures/')

    print(f"质量: {info['mass']} kg")
    print(f"弹簧常数: {info['spring_constant']} N/m")
    print(f"自然长度: {info['natural_length']} m") # 打印自然长度
    print(f"自然频率: {info['natural_frequency']:.3f} rad/s")
    print(f"特征值: {info['eigenvalues']}")


if __name__ == "__main__":
    main()
 