"""
可视化工具 - 绘制仿真结果和动画
修正版：移除了墙和连接墙的弹簧，以符合 B.2.13 空间双质量结构的定义
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import Dict, Any, Optional
from matplotlib.gridspec import GridSpec

class Animator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig_size = config['visualization']['figure_size']
        self.dpi = config['visualization']['dpi']
        self.mass_size = config['physical']['mass_width']  # 物块大小=宽度

        
    def plot_results(self, results: Dict[str, np.ndarray], 
                    reference: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        绘制仿真结果（静态曲线图）
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size, dpi=self.dpi)
        time = results['time']
        
        # 位移曲线
        axes[0, 0].plot(time, results['x1'], label='Mass 1', linewidth=2)
        axes[0, 0].plot(time, results['x2'], label='Mass 2', linewidth=2)
        if reference is not None:
            axes[0, 0].plot(time, reference['x1'], '--', label='Ref x1', alpha=0.7)
            axes[0, 0].plot(time, reference['x2'], '--', label='Ref x2', alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Displacement')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 速度曲线
        axes[0, 1].plot(time, results['v1'], label='Mass 1', linewidth=2)
        axes[0, 1].plot(time, results['v2'], label='Mass 2', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 控制输入
        axes[1, 0].plot(time, results['control'], linewidth=2, color='red')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Force (N)')
        axes[1, 0].set_title('Control Input')
        axes[1, 0].grid(True)
        
        # 扰动
        axes[1, 1].plot(time, results['disturbances'][:, 0], label='w1', linewidth=2)
        axes[1, 1].plot(time, results['disturbances'][:, 1], label='w2', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Disturbance Force (N)')
        axes[1, 1].set_title('Disturbances')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_spring(self, x_start: float, x_end: float, 
                     y_center: float = 0.0, num_coils: int = 8, 
                     height: float = 0.15) -> np.ndarray:
        """
        生成弹簧的锯齿状路径点
        """
        x = np.linspace(x_start, x_end, num_coils * 4 + 2)
        y = np.zeros_like(x)
        y[0] = y_center
        y[-1] = y_center
        
        for i in range(1, len(x) - 1):
            coil_pos = (i - 1) % 4
            if coil_pos == 0:
                y[i] = y_center + height
            elif coil_pos == 1:
                y[i] = y_center - height
            elif coil_pos == 2:
                y[i] = y_center + height
            else:
                y[i] = y_center - height
        
        y[0] = y_center
        y[-1] = y_center
        
        return np.column_stack([x, y])
    
    def animate_system(self, results: Dict[str, np.ndarray]) -> None:
        """
        创建系统动画 - 符合 B.2.13: 自由空间中的两个质量块
        """
        # 创建图形，使用GridSpec进行布局
        fig = plt.figure(figsize=(14, 8), dpi=self.dpi)
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
        
        # 左侧：物理系统动画（跨越所有行）
        ax_anim = fig.add_subplot(gs[:, 0])
        
        # 右侧上部：控制输入
        ax_control = fig.add_subplot(gs[0, 1])
        
        # 右侧中部：位移曲线
        ax_disp = fig.add_subplot(gs[1, 1])
        
        # 右侧下部：速度曲线
        ax_vel = fig.add_subplot(gs[2, 1])
        
        # ========== 设置左侧物理系统动画 ==========
        # 计算显示范围
        x_data = np.concatenate([results['x1'], results['x2']])
        x_min, x_max = np.min(x_data) - 1.0, np.max(x_data) + 1.5
        
        ax_anim.set_xlim(x_min, x_max)
        ax_anim.set_ylim(-0.8, 0.8)
        ax_anim.set_aspect('equal')
        ax_anim.grid(True, alpha=0.3)
        ax_anim.set_title('B.2.13 Two-Mass-Spring System (Free Space)', fontsize=14, fontweight='bold')
        ax_anim.set_xlabel('Position (m)')
        ax_anim.set_ylabel('y (m)')
        
        # 【修改点】移除了墙的绘制代码
        # 【修改点】移除了地面的绘制代码
        
        # 创建正方形质量块
        mass1_patch = FancyBboxPatch(
            (results['x1'][0] - self.mass_size/2, -self.mass_size/2),
            self.mass_size, self.mass_size,
            boxstyle="round,pad=0.02",
            fc='#3498db', ec='black', linewidth=2,
            alpha=0.9
        )
        mass2_patch = FancyBboxPatch(
            (results['x2'][0] - self.mass_size/2, -self.mass_size/2),
            self.mass_size, self.mass_size,
            boxstyle="round,pad=0.02",
            fc='#e74c3c', ec='black', linewidth=2,
            alpha=0.9
        )
        
        # 【修改点】只保留一根弹簧（连接两个质量块），移除连接墙的弹簧
        spring_line, = ax_anim.plot([], [], 'k-', linewidth=2.5)
        
        # 质量块标签
        mass1_text = ax_anim.text(0, 0, 'M1', ha='center', va='center', 
                                  color='white', fontweight='bold', fontsize=10)
        mass2_text = ax_anim.text(0, 0, 'M2', ha='center', va='center', 
                                  color='white', fontweight='bold', fontsize=10)
        
        # 时间和状态文本
        time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, 
                                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        state_text = ax_anim.text(0.02, 0.02, '', transform=ax_anim.transAxes,
                                  fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        ax_anim.add_patch(mass1_patch)
        ax_anim.add_patch(mass2_patch)
        
        # ========== 设置右侧曲线图 ==========
        time_array = results['time']
        
        # 控制输入曲线
        ax_control.plot(time_array, results['control'], color='red', linewidth=1.5, alpha=0.6)
        ax_control.set_ylabel('Control (N)', color='red')
        ax_control.set_xlim(0, time_array[-1])
        ax_control.set_ylim(np.min(results['control']) - 0.5, np.max(results['control']) + 0.5)
        ax_control.grid(True, alpha=0.3)
        ax_control.set_title('Control Input', fontsize=11)
        control_dot, = ax_control.plot([], [], 'ro', markersize=8)
        control_line, = ax_control.plot([], [], 'r-', linewidth=2.5)
        
        # 位移曲线
        ax_disp.plot(time_array, results['x1'], color='#3498db', linewidth=1.5, alpha=0.6, label='M1')
        ax_disp.plot(time_array, results['x2'], color='#e74c3c', linewidth=1.5, alpha=0.6, label='M2')
        ax_disp.set_ylabel('Position (m)')
        ax_disp.set_xlim(0, time_array[-1])
        y_min = min(np.min(results['x1']), np.min(results['x2']))
        y_max = max(np.max(results['x1']), np.max(results['x2']))
        ax_disp.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_disp.grid(True, alpha=0.3)
        ax_disp.set_title('Displacement', fontsize=11)
        ax_disp.legend(loc='upper right', fontsize=8)
        disp1_dot, = ax_disp.plot([], [], 'o', color='#3498db', markersize=8)
        disp2_dot, = ax_disp.plot([], [], 'o', color='#e74c3c', markersize=8)
        disp1_line, = ax_disp.plot([], [], '-', color='#3498db', linewidth=2.5)
        disp2_line, = ax_disp.plot([], [], '-', color='#e74c3c', linewidth=2.5)
        
        # 速度曲线
        ax_vel.plot(time_array, results['v1'], color='#2ecc71', linewidth=1.5, alpha=0.6, label='M1')
        ax_vel.plot(time_array, results['v2'], color='#f39c12', linewidth=1.5, alpha=0.6, label='M2')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_xlim(0, time_array[-1])
        y_min_v = min(np.min(results['v1']), np.min(results['v2']))
        y_max_v = max(np.max(results['v1']), np.max(results['v2']))
        ax_vel.set_ylim(y_min_v - 0.1, y_max_v + 0.1)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.set_title('Velocity', fontsize=11)
        ax_vel.legend(loc='upper right', fontsize=8)
        vel1_dot, = ax_vel.plot([], [], 'o', color='#2ecc71', markersize=8)
        vel2_dot, = ax_vel.plot([], [], 'o', color='#f39c12', markersize=8)
        vel1_line, = ax_vel.plot([], [], '-', color='#2ecc71', linewidth=2.5)
        vel2_line, = ax_vel.plot([], [], '-', color='#f39c12', linewidth=2.5)
        
        plt.tight_layout()
        
        # ========== 动画更新函数 ==========
        def init():
            """初始化动画"""
            mass1_patch.set_x(results['x1'][0] - self.mass_size/2)
            mass2_patch.set_x(results['x2'][0] - self.mass_size/2)
            spring_line.set_data([], [])
            mass1_text.set_position((results['x1'][0], 0))
            mass2_text.set_position((results['x2'][0], 0))
            time_text.set_text('')
            state_text.set_text('')
            
            # 曲线图
            control_dot.set_data([], [])
            control_line.set_data([], [])
            disp1_dot.set_data([], [])
            disp2_dot.set_data([], [])
            disp1_line.set_data([], [])
            disp2_line.set_data([], [])
            vel1_dot.set_data([], [])
            vel2_dot.set_data([], [])
            vel1_line.set_data([], [])
            vel2_line.set_data([], [])
            
            return (mass1_patch, mass2_patch, spring_line,
                    mass1_text, mass2_text, time_text, state_text,
                    control_dot, control_line, disp1_dot, disp2_dot,
                    disp1_line, disp2_line, vel1_dot, vel2_dot,
                    vel1_line, vel2_line)
        
        def update(frame):
            """更新动画帧"""
            current_time = time_array[frame]
            x1, x2 = results['x1'][frame], results['x2'][frame]
            
            # 更新正方形位置
            mass1_patch.set_x(x1 - self.mass_size/2)
            mass2_patch.set_x(x2 - self.mass_size/2)
            
            # 更新标签位置
            mass1_text.set_position((x1, 0))
            mass2_text.set_position((x2, 0))
            
            # 【修改点】只更新连接 M1 和 M2 的弹簧
            spring_points = self._draw_spring(x1 + self.mass_size/2, x2 - self.mass_size/2, 0, 10, 0.1)
            spring_line.set_data(spring_points[:, 0], spring_points[:, 1])
            
            # 更新文本信息
            time_text.set_text(f'Time: {current_time:.2f} s')
            state_text.set_text(f'x1: {x1:.3f} m\nx2: {x2:.3f} m\nv1: {results["v1"][frame]:.3f} m/s\nv2: {results["v2"][frame]:.3f} m/s')
            
            # 更新曲线图
            control_line.set_data(time_array[:frame], results['control'][:frame])
            control_dot.set_data([current_time], [results['control'][frame]])
            
            disp1_line.set_data(time_array[:frame], results['x1'][:frame])
            disp2_line.set_data(time_array[:frame], results['x2'][:frame])
            disp1_dot.set_data([current_time], [results['x1'][frame]])
            disp2_dot.set_data([current_time], [results['x2'][frame]])
            
            vel1_line.set_data(time_array[:frame], results['v1'][:frame])
            vel2_line.set_data(time_array[:frame], results['v2'][:frame])
            vel1_dot.set_data([current_time], [results['v1'][frame]])
            vel2_dot.set_data([current_time], [results['v2'][frame]])
            
            return (mass1_patch, mass2_patch, spring_line,
                    mass1_text, mass2_text, time_text, state_text,
                    control_dot, control_line, disp1_dot, disp2_dot,
                    disp1_line, disp2_line, vel1_dot, vel2_dot,
                    vel1_line, vel2_line)
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, update, frames=len(time_array),
            init_func=init, blit=True,
            interval=self.config['visualization']['animation_interval']
        )
        
        plt.show()
        
        # 保存动画
        if self.config['visualization']['save_animation']:
            filename = self.config['visualization']['animation_filename']
            anim.save(filename, writer='ffmpeg', fps=30, dpi=100)
            print(f"动画已保存为: {filename}")
