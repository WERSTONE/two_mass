"""
MPC Controller Performance Analysis and Visualization - Publication Quality
Advanced visualization for controller performance evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import fft, signal
import matplotlib as mpl

# Set publication quality parameters
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.unicode_minus': False,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'axes.grid': True,
})

# Academic color scheme (colorblind-friendly)
COLORS = {
    'mass1': '#1f77b4',      # Blue
    'mass2': '#d62728',      # Red
    'vibration': '#2ca02c',  # Green
    'control': '#ff7f0e',     # Orange
    'disturbance': '#9467bd', # Purple
    'reference': '#17becf',  # Cyan
    'error': '#e377c2',      # Pink
    'energy': '#bcbd22',      # Olive
    'settling': '#8c564b'    # Brown
}

def plot_performance_indicators(results, config, plant_info):
    """
    绘制详细的性能指标分析图：
    1. 跟踪误差 (Tracking Error)
    2. 控制能量累积 (Control Energy)
    3. 柔性振动量 (Vibration/Spring Deformation)
    """
    time = results['time']
    x2 = results['x2']
    u = results['control']
    x1 = results['x1']
    
    # 目标位置 (假设恒定)
    target = config['initial_conditions']['x2'] 
    
    # --- 计算指标 ---
    # 1. 误差
    error = x2 - target
    
    # 2. RMSE (只计算后半段稳态，避免初始波动影响)
    steady_idx = int(len(time) * 0.5)
    rmse = np.sqrt(np.mean(error[steady_idx:]**2))
    
    # 3. 能量 (积分 u^2)
    dt = time[1] - time[0]
    energy_cumulative = np.cumsum(u**2) * dt
    total_energy = energy_cumulative[-1]
    
    # 4. 相对位移 (弹簧形变)
    deformation = x2 - x1
    
    # --- 绘图 ---
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.4)
    
    # Subplot 1: Tracking Error
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, error, color='#d62728', label='Error $(x_2 - x_{ref})$')
    ax1.fill_between(time, error, 0, alpha=0.1, color='#d62728')
    ax1.set_title(f'Tracking Error Analysis (Steady State RMSE = {rmse:.4f} m)', fontweight='bold')
    ax1.set_ylabel('Error [m]')
    ax1.grid(True)
    ax1.legend()
    
    # Subplot 2: Control Energy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, energy_cumulative, color='#ff7f0e', label='$\int u^2 dt$')
    ax2.set_title(f'Control Effort Accumulation (Total Energy = {total_energy:.2f} $N^2s$)', fontweight='bold')
    ax2.set_ylabel('Energy')
    ax2.grid(True)
    ax2.legend()
    
    # Subplot 3: Structural Vibration (Deformation)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, deformation, color='#2ca02c', label='Spring Deformation $(x_2 - x_1)$')
    # 画出自然长度参考线
    natural_len = plant_info.get('natural_length', 1.0) # 假设默认1.0
    ax3.axhline(natural_len, color='k', linestyle='--', alpha=0.5, label='Natural Length')
    ax3.set_title('Flexible Structure Vibration', fontweight='bold')
    ax3.set_ylabel('Distance [m]')
    ax3.set_xlabel('Time [s]')
    ax3.legend()
    
    return fig


def calculate_settling_time_regulation(response, time, target=1.0, threshold=0.02):
    """
    Calculate settling time for regulation problem (targeting a specific setpoint).
    Finds time after which the response stays within target +/- threshold.
    """
    lower_bound = target * (1 - threshold)
    upper_bound = target * (1 + threshold)
    
    # 从后往前找，找到最后一个超出误差带的点
    for i in range(len(response)-1, -1, -1):
        if response[i] < lower_bound or response[i] > upper_bound:
            if i < len(time) - 1:
                return time[i+1]
    return time[-1]


def calculate_steady_state_error(response, target=1.0):
    """Calculate final steady state error (SSE)"""
    # 使用最后100个点的平均值作为最终状态
    final_value = np.mean(response[-100:])
    return final_value - target


def plot_time_response_with_metrics(results, config, plant_info):
    """Time-domain response with comprehensive metrics annotation"""
    time = results['time']
    x1 = results['x1']
    x2 = results['x2']
    control = results['control']
    
    # 目标设定点
    target_ref = 1.0
    
    # 使用1行2列的横排布局
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    
    # 1. Displacement response
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, x2, color=COLORS['mass2'], linewidth=2.5, label='Mass 2 (Output)')
    ax1.plot(time, x1, color=COLORS['mass1'], linewidth=1.8, alpha=0.7, label='Mass 1 (Actuator)')
    ax1.axhline(y=target_ref, color=COLORS['reference'], linestyle='--', linewidth=1.5, alpha=0.8, label='Reference')
    
    # 计算干扰抑制场景下的指标
    error = x2 - target_ref
    abs_error = np.abs(error)
    
    # 1. 最大偏差: 对应干扰强度
    max_dev = np.max(abs_error)
    max_dev_percent = (max_dev / target_ref) * 100
    t_max_dev = time[np.argmax(abs_error)]
    
    # 2. 稳态误差
    sse = calculate_steady_state_error(x2, target_ref)
    
    # 3. 调节时间 (2% 误差带)
    settling_time = calculate_settling_time_regulation(x2, time, target_ref, 0.02)
    
    # 4. 恢复时间: 从最大偏差恢复到最大偏差的10%所需的时间
    idx_peak = np.argmax(abs_error)
    recovery_threshold = 0.1 * max_dev
    # 寻找峰值之后，误差第一次小于阈值的时刻
    recovery_indices = np.where((abs_error[idx_peak:] <= recovery_threshold))[0]
    
    if len(recovery_indices) > 0:
        recovery_idx = idx_peak + recovery_indices[0]
        recovery_time = time[recovery_idx] - t_max_dev
    else:
        recovery_time = np.nan # 如果在仿真时间内未恢复，则为NaN

    # 控制量指标
    max_control = np.max(np.abs(control))
    rms_control = np.sqrt(np.mean(control**2))
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True)
    
    # 2. Control input
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, control, color=COLORS['control'], linewidth=2.0, label='Control Input')
    ax2.fill_between(time, 0, control, alpha=0.2, color=COLORS['control'])
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Force (N)')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True)
    
    # 在控制台输出更符合干扰抑制场景的指标
    print("\n" + "="*50)
    print("Time-Domain Performance Metrics (Regulation)")
    print("="*50)
    print(f"  Max Deviation:      {max_dev_percent:>8.2f} % (at {t_max_dev:.3f}s)")
    print(f"  Recovery Time:      {recovery_time:>8.3f} s (to 10% error)")
    print(f"  Settling Time:      {settling_time:>8.3f} s (2% band)")
    print(f"  Steady State Err:   {sse:>8.4f} m")
    print("-" * 50)
    print(f"  Max Control Force:  {max_control:>8.3f} N")
    print(f"  RMS Control Force:  {rms_control:>8.3f} N")
    print("="*50)
    
    return fig


def plot_frequency_domain_analysis(results, config, plant_info):
    """
    频域分析图（美化版）
    反映控制器的频率响应和滤波特性
    """
    time = results["time"]
    x2 = results["x2"]
    control = results["control"]
    
    dt = time[1] - time[0]
    fs = 1.0 / dt
    n = len(time)
    
    # FFT计算
    freq = fft.fftfreq(n, dt)
    freq = freq[:n//2]  # 只取正频率
    
    x2_fft = fft.fft(x2)
    x2_mag = np.abs(x2_fft)[:n//2] / n * 2  # 归一化
    
    control_fft = fft.fft(control)
    control_mag = np.abs(control_fft)[:n//2] / n * 2
    
    # 计算PSD（功率谱密度）
    x2_psd = np.abs(x2_fft[:n//2])**2 / (fs * n)
    control_psd = np.abs(control_fft[:n//2])**2 / (fs * n)
    
    # 转换为dB
    x2_psd_db = 10 * np.log10(x2_psd + 1e-10)
    control_psd_db = 10 * np.log10(control_psd + 1e-10)
    
    # 计算频域指标
    # 忽略直流分量 (0 Hz)，寻找主导频率
    idx_ignore_dc = 1 
    
    if idx_ignore_dc < len(freq):
        peak_freq_x2 = freq[idx_ignore_dc + np.argmax(x2_mag[idx_ignore_dc:])]
        peak_freq_control = freq[idx_ignore_dc + np.argmax(control_mag[idx_ignore_dc:])]
        peak_val_x2 = np.max(x2_mag[idx_ignore_dc:])
        peak_val_control = np.max(control_mag[idx_ignore_dc:])
    else:
        peak_freq_x2 = 0.0
        peak_freq_control = 0.0
        peak_val_x2 = 0.0
        peak_val_control = 0.0
    
    # 调整图形大小和布局
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. 位置x2的频谱（美化版）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(freq, x2_mag, color=COLORS["mass2"], linewidth=2.0, alpha=0.9)
    # 添加半透明填充
    ax1.fill_between(freq, 0, x2_mag, color=COLORS["mass2"], alpha=0.15)
    # 【修改】只保留虚线，删除箭头标记
    ax1.axvline(x=peak_freq_x2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'Peak: {peak_freq_x2:.3f} Hz')
    ax1.set_xlabel("Frequency (Hz)", fontweight='bold')
    ax1.set_ylabel("Amplitude", fontweight='bold')
    ax1.set_title("Position x2 Frequency Spectrum", fontweight='bold', fontsize=13)
    ax1.grid(True, which="both", alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # 2. 控制输入的频谱（美化版）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(freq, control_mag, color=COLORS["control"], linewidth=2.0, alpha=0.9)
    # 添加半透明填充
    ax2.fill_between(freq, 0, control_mag, color=COLORS["control"], alpha=0.15)
    # 【修改】只保留虚线，删除箭头标记
    ax2.axvline(x=peak_freq_control, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Peak: {peak_freq_control:.3f} Hz')
    # 添加高频噪声阈值线
    ax2.axvline(x=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Noise Cutoff')
    ax2.set_xlabel("Frequency (Hz)", fontweight='bold')
    ax2.set_ylabel("Amplitude", fontweight='bold')
    ax2.set_title("Control Input Frequency Spectrum", fontweight='bold', fontsize=13)
    ax2.grid(True, which="both", alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # 3. 位置PSD（美化版）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(freq, x2_psd_db, color=COLORS["mass2"], linewidth=2.0, alpha=0.9)
    # 添加半透明填充（对数坐标下需要特殊处理）
    ax3.fill_between(freq, x2_psd_db.min(), x2_psd_db, color=COLORS["mass2"], alpha=0.15)
    # 标记-20dB线（表示高频衰减良好）
    ax3.axhline(y=-20, color='green', linestyle='--', linewidth=1.5, alpha=0.6, 
                label='-20 dB Threshold')
    ax3.set_xlabel("Frequency (Hz)", fontweight='bold')
    ax3.set_ylabel("Power Spectral Density (dB)", fontweight='bold')
    ax3.set_title("Position x2 PSD (Power Distribution)", fontweight='bold', fontsize=13)
    ax3.grid(True, which="both", alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', framealpha=0.9)
    
    # 4. 控制PSD（美化版）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(freq, control_psd_db, color=COLORS["control"], linewidth=2.0, alpha=0.9)
    # 添加半透明填充
    ax4.fill_between(freq, control_psd_db.min(), control_psd_db, color=COLORS["control"], alpha=0.15)
    # 标记-20dB线
    ax4.axhline(y=-20, color='green', linestyle='--', linewidth=1.5, alpha=0.6,
                label='-20 dB Threshold')
    # 标记噪声截止频率
    noise_cutoff_idx = np.where(control_psd_db < -20)[0]
    if len(noise_cutoff_idx) > 0:
        noise_cutoff = freq[noise_cutoff_idx[0]]
        ax4.axvline(x=noise_cutoff, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                    label=f'Cutoff: {noise_cutoff:.2f} Hz')
    ax4.set_xlabel("Frequency (Hz)", fontweight='bold')
    ax4.set_ylabel("Power Spectral Density (dB)", fontweight='bold')
    ax4.set_title("Control Input PSD (Power Distribution)", fontweight='bold', fontsize=13)
    ax4.grid(True, which="both", alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right', framealpha=0.9)
    
    plt.suptitle("Frequency Domain Analysis: Amplitude Spectrum & Power Distribution", 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 输出频域指标
    print("\n" + "="*50)
    print("Frequency-Domain Metrics")
    print("="*50)
    print(f"  Dominant Freq (Pos): {peak_freq_x2:>8.3f} Hz")
    print(f"  Peak Amplitude (Pos): {peak_val_x2:>8.4f}")
    print(f"  Dominant Freq (Ctrl):{peak_freq_control:>8.3f} Hz")
    print(f"  Peak Amplitude (Ctrl):{peak_val_control:>8.4f}")
    print("="*50)
    
    return fig


def plot_phase_portrait_and_stability(results, config, plant_info):
    """
    相平面图和稳定性分析（仅保留相平面图）
    反映系统的动态特性和控制器的稳定性
    """
    x1 = results["x1"]
    x2 = results["x2"]
    v1 = results["v1"]
    v2 = results["v2"]
    time = results["time"]
    
    # 只使用1行2列，只保留两个相平面图
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # 1. 质量块2的相平面图
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 使用颜色编码表示时间
    scatter = ax1.scatter(x2 - 1, v2, c=time, cmap="viridis", s=8, alpha=0.7, edgecolors='none')
    ax1.set_xlabel("Position Error x2 - 1 (m)", fontweight='bold')
    ax1.set_ylabel("Velocity v2 (m/s)", fontweight='bold')
    ax1.set_title("Phase Portrait: Mass 2 (Output)", fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Time (s)", fontweight='bold')
    
    # 标记起点和终点（美化）
    ax1.plot((x2[0] - 1), v2[0], "o", color='lime', markersize=14, 
             markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=5)
    ax1.plot((x2[-1] - 1), v2[-1], "o", color='red', markersize=14, 
             markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=5)
    # 添加坐标原点标记
    ax1.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Equilibrium', zorder=4)
    ax1.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # 2. 质量块1的相平面图
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(x1, v1, c=time, cmap="viridis", s=8, alpha=0.7, edgecolors='none')
    ax2.set_xlabel("Position x1 (m)", fontweight='bold')
    ax2.set_ylabel("Velocity v1 (m/s)", fontweight='bold')
    ax2.set_title("Phase Portrait: Mass 1 (Actuator)", fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Time (s)", fontweight='bold')
    
    ax2.plot(x1[0], v1[0], "o", color='lime', markersize=14, 
             markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=5)
    ax2.plot(x1[-1], v1[-1], "o", color='red', markersize=14, 
             markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=5)
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)
    
    plt.suptitle("Phase Portrait Analysis: System Stability & Convergence", 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    return fig


def generate_publication_plots(results, config, plant_info, output_dir='./'):
    """Generate all publication-quality plots"""
    print("Generating publication-quality MPC control performance plots...")

    # 自动创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    fig1 = plot_time_response_with_metrics(results, config, plant_info)
    fig2 = plot_frequency_domain_analysis(results,config,plant_info)
    fig3 = plot_phase_portrait_and_stability(results,config,plant_info)
    
    fig_metrics = plot_performance_indicators(results, config, plant_info)
    fig_metrics.savefig(f'{output_dir}/control_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig_metrics)
    print("  ✓ Control performance metrics - DONE")
    # 保存 fig1
    if fig1:
        fig1.savefig(f'{output_dir}/time_response_metrics.png', dpi=300,
                   bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        print("  ✓ Time response with metrics - DONE")

    # 保存 fig2
    if fig2:
        fig2.savefig(f'{output_dir}/frequency_domain_analysis.png', dpi=300,
                   bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print("  ✓ Frequency domain analysis - DONE")

    # 保存 fig3
    if fig3:
        fig3.savefig(f'{output_dir}/phase_portrait_stability.png', dpi=300,
                   bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        print("  ✓ Phase portrait and stability - DONE")
