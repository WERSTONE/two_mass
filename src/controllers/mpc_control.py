"""
MPC 控制器 - 基于完整状态空间模型的模型预测控制
新增功能：集成扰动观测器 (ESO) + 热启动 + 软碰撞约束（非严格不等式）
"""

import numpy as np
from scipy.linalg import block_diag, expm
from src.controllers.base_controller import BaseController
from typing import Dict, Any, Optional
import cvxpy as cp  # 用于二次规划求解

class MPCController(BaseController):
    """模型预测控制器 (支持弹簧自然长度 + 扰动观测器 + 软碰撞约束)"""
    
    def __init__(self, config: Dict[str, Any], initial_state: np.ndarray = None):
        super().__init__(config)
        
        # --- 1. 系统参数 ---
        self.m = 1.0
        self.k = 1.0
        self.dt = 0.01  # 采样时间
        self.natural_length = config.get('natural_length', 0.0)
        self.mass_width = config.get('mass_width', 0.3)  # 物块宽度（用于碰撞约束）
        
        # --- 【新增】软约束参数 ---
        self.collision_penalty_weight = config.get('collision_penalty_weight', 1000.0)  # 碰撞惩罚权重
        
        # --- 【新增】ESO 观测器参数 ---
        self.omega_o = config.get('eso_bandwidth', 50.0)  # 观测器带宽
        self.beta_1 = 3.0 * self.omega_o
        self.beta_2 = 3.0 * (self.omega_o ** 2)
        self.beta_3 = (self.omega_o ** 3)
        
        # 【新增】扰动估计限幅值（防止峰值传递）
        self.w_max = config.get('w_max', 10.0)  # 从配置读取，默认10.0
        
        # 【新增】热启动时间（从配置读取，默认1.0秒）
        self.ramp_time = config.get('ramp_time', 1.0)  # 热启动持续时间

        # --- 【新增】控制力约束参数 ---
        self.u_max = config.get('u_max', 10.0)
        self.u_min = config.get('u_min', -10.0)
        
        # --- 【核心修改】初始化 ESO 状态 ---
        # 获取初始状态，如果未传入则默认为全0
        if initial_state is None:
            initial_state = np.zeros(4)
            
        x1_init, x2_init, v1_init, v2_init = initial_state
        
        # 初始化 ESO 状态 [z1(位置), z2(速度), z3(扰动加速度)]
        # 分别针对 Mass 1 和 Mass 2
        # 【修复】使用真实的初始位置和速度，而不是 0。z3 初始化为 0（假设初始无未知扰动）
        self.eso_state_m1 = np.array([x1_init, v1_init, 0.0]) 
        self.eso_state_m2 = np.array([x2_init, v2_init, 0.0])
        
        # --- 2. 连续时间状态空间矩阵 ---
        # 状态 x = [x1, x2, v1, v2]^T
        self.A_cont = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-self.k/self.m, self.k/self.m, 0, 0],
            [self.k/self.m, -self.k/self.m, 0, 0]
        ])
        
        self.B_cont = np.array([
            [0],
            [0],
            [1/self.m],
            [0]
        ])
        
        # 扰动输入矩阵 (将力 w 转换为加速度)
        # w1 作用于 m1, w2 作用于 m2
        self.B_w_cont = np.array([
            [0, 0],
            [0, 0],
            [1/self.m, 0],
            [0, 1/self.m]
        ])
        
        # --- 3. 离散化 (包含常数项 d 和 扰动矩阵 Bw) ---
        # 常数加速度扰动 d_cont = [0, 0, -k*L0/m, k*L0/m]^T
        self.d_cont = np.array([
            0.0,
            0.0,
            -self.k/self.m * self.natural_length,
            self.k/self.m * self.natural_length
        ])
        
        self.Ad, self.Bd, self.d_disc = self._discretize_zoh_with_const(
            self.A_cont, self.B_cont, self.d_cont, self.dt
        )
        
        # 离散化扰动矩阵 Bw
        self.Bw_disc = self._discretize_Bw(self.B_w_cont, self.dt)

        # --- 4. MPC 参数 ---
        self.N = config.get('prediction_horizon', 20) 
        
        # 代价函数权重
        q_x1 = config.get('q_x1', 0)
        q_x2 = config.get('q_x2', 100.0)
        q_v1 = config.get('q_v1', 0)
        q_v2 = config.get('q_v2', 10.0)
        self.Q = np.diag([q_x1, q_x2, q_v1, q_v2])
        
        self.R = np.array([[config.get('r_input', 0.01)]])
        
        # --- 5. 预计算 MPC 矩阵 ---
        self._build_mpc_matrices()


        
    def _discretize_Bw(self, Bw_cont, dt):
        """离散化扰动输入矩阵"""
        return self.Ad @ Bw_cont * dt

    def _discretize_zoh_with_const(self, A, B, d, dt):
        """
        零阶保持离散化，同时离散化状态方程 x' = Ax + Bu + d
        使用增广矩阵方法进行精确离散化
        """
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        
        # 构造增广矩阵 M = [A B d; 0 0 0]
        M = np.zeros((n_states + n_inputs + 1, n_states + n_inputs + 1))
        M[:n_states, :n_states] = A
        M[:n_states, n_states:n_states+n_inputs] = B
        M[:n_states, -1] = d  # 最后一列为常数扰动项
        
        Phi = expm(M * dt)
        
        Ad = Phi[:n_states, :n_states]
        Bd = Phi[:n_states, n_states:n_states+n_inputs]
        dd = Phi[:n_states, -1]  # 离散化后的常数项影响
        
        return Ad, Bd, dd

    def _build_mpc_matrices(self):
        """
        构建预测所需的常数矩阵
        """
        n_states = self.Ad.shape[0]
        n_inputs = self.Bd.shape[1]
        
        # 1. 构造块对角矩阵 Q_bar 和 R_bar
        Q_list = [self.Q] * self.N
        self.Q_bar = block_diag(*Q_list)
        
        R_list = [self.R] * self.N
        self.R_bar = block_diag(*R_list)
        
        # 2. 构造状态预测矩阵 Sx (N*n_states x n_states)
        power_A = np.eye(n_states)
        Sx_rows = []
        for _ in range(self.N):
            power_A = self.Ad @ power_A
            Sx_rows.append(power_A)
        self.Sx = np.vstack(Sx_rows)
        
        # 3. 构造输入影响矩阵 Su (N*n_states x N*n_inputs)
        Su_rows = []
        power_A = np.eye(n_states) # A^0 = I
        for i in range(self.N):
            row = np.zeros((n_states, self.N * n_inputs))
            for j in range(i + 1):
                if i == j:
                    term = self.Bd
                else:
                    p = i - j
                    mat_p = np.eye(n_states)
                    for _ in range(p):
                        mat_p = self.Ad @ mat_p
                    term = mat_p @ self.Bd
                
                col_start = j * n_inputs
                row[:, col_start : col_start + n_inputs] = term
            Su_rows.append(row)
        self.Su = np.vstack(Su_rows)
        
        # 4. 计算常数项（自然长度）的累积影响
        self.total_const_offset = np.zeros((self.N * n_states, 1))
        current_offset = np.zeros((n_states, 1))
        
        for i in range(self.N):
            if i == 0:
                current_offset = self.d_disc.reshape(-1, 1)
            else:
                current_offset = self.Ad @ current_offset + self.d_disc.reshape(-1, 1)
            
            start_row = i * n_states
            end_row = (i + 1) * n_states
            self.total_const_offset[start_row:end_row, :] = current_offset
            
        # --- 【修复】5. 计算扰动的影响矩阵 Sw ---
        # 扰动在未来 N 步内持续存在，因此需要计算累积影响 sum(A^j * Bw)
        # 形状: (N * n_states, 2)
        self.Sw = np.zeros((self.N * n_states, 2))
        
        # 累积项初始化
        cumulative_disturbance_effect = np.zeros((n_states, 2))
        # 当前时刻的扰动直接影响项 (A^0 * Bw)
        current_term = np.copy(self.Bw_disc)
        
        for i in range(self.N):
            # 累加到当前步的影响: Sum_{j=0}^{i} A^j * Bw
            cumulative_disturbance_effect = cumulative_disturbance_effect + current_term
            
            start_row = i * n_states
            end_row = (i + 1) * n_states
            self.Sw[start_row:end_row, :] = cumulative_disturbance_effect
            
            # 更新下一项为 A * current_term (即 A^{i+1} * Bw)
            current_term = self.Ad @ current_term
            
        # 6. 预计算 Hessian 矩阵 H (使用 Cholesky 分解预计算因子以提高稳定性)
        self.H = 2 * (self.Su.T @ self.Q_bar @ self.Su + self.R_bar)
        try:
            from scipy.linalg import cho_factor, cho_solve
            self.H_factor = cho_factor(self.H, check_finite=False)
            self.use_cholesky = True
        except:
            self.H_inv = np.linalg.inv(self.H)
            self.use_cholesky = False
        
        self.Su_T_Q = self.Su.T @ self.Q_bar

    def _update_eso(self, x_meas, u_curr):
        """
        更新线性扩张状态观测器 (LESO)
        x_meas: 当前测量位置 [x1, x2]
        u_curr: 当前控制输入 u (作用在 m1)
        """
        x1_m, x2_m = x_meas[0], x_meas[1]
        
        # --- 更新 Mass 1 的 ESO ---
        # z1: pos, z2: vel, z3: total_disturbance_acc (acceleration)
        # 模型: z1' = z2, z2' = f_total + b0*u
        # f_total 包含了弹簧力、自然长度力和外部扰动 w1
        z1_1, z2_1, z3_1 = self.eso_state_m1
        e1 = x1_m - z1_1 # 观测误差
        
        # 离散化 ESO 更新 (欧拉法)
        z1_1_new = z1_1 + self.dt * (z2_1 + self.beta_1 * e1)
        z2_1_new = z2_1 + self.dt * (z3_1 + u_curr/self.m + self.beta_2 * e1)
        z3_1_new = z3_1 + self.dt * (self.beta_3 * e1)
        
        self.eso_state_m1 = np.array([z1_1_new, z2_1_new, z3_1_new])
        
        # --- 更新 Mass 2 的 ESO ---
        # Mass 2 没有直接控制输入 u
        z1_2, z2_2, z3_2 = self.eso_state_m2
        e2 = x2_m - z1_2
        
        z1_2_new = z1_2 + self.dt * (z2_2 + self.beta_1 * e2)
        z2_2_new = z2_2 + self.dt * (z3_2 + self.beta_2 * e2)
        z3_2_new = z3_2 + self.dt * (self.beta_3 * e2)
        
        self.eso_state_m2 = np.array([z1_2_new, z2_2_new, z3_2_new])
        
        # --- 提取外部扰动力估计 ---
        # ESO 估计的 z3 是总加速度扰动 (f_total)
        # f_total_m1 = k(x2-x1-L0)/m + w1/m
        # f_total_m2 = -k(x2-x1-L0)/m + w2/m
        spring_acc = self.k * (x2_m - x1_m - self.natural_length) / self.m
        
        w1_est = (z3_1_new - spring_acc) * self.m
        w2_est = (z3_2_new - (-spring_acc)) * self.m
        
        return np.array([w1_est, w2_est])

    def control(self, state: np.ndarray, 
                reference: Optional[np.ndarray] = None,
                time: float = 0.0) -> float:
        """计算 MPC 控制输入（修正版：包含输入约束变换与最终抗饱和）"""
        
        # 1. 运行 ESO：必须传入【上一时刻】的控制量 self.last_u
        u_prev = getattr(self, 'last_u', 0.0)
        w_est = self._update_eso(state, u_prev)
        
        # 2. 扰动估计限幅 (w_max)
        w_est = np.clip(w_est, -self.w_max, self.w_max)
        
        # 3. 热启动 (Ramp Factor)
        ramp_factor = min(time / self.ramp_time, 1.0) if self.ramp_time > 0 else 1.0
        w_est_safe = w_est * ramp_factor
        
        # 4. 确定参考轨迹
        x_ref = np.zeros((self.N * 4, 1))
        ref_x2 = reference[1] if reference is not None and len(reference) > 1 else 0.0
        for i in range(self.N):
            x_ref[i * 4 + 1] = ref_x2

        # 5. 构造 QP 问题
        x0 = state.reshape(-1, 1)
        U = cp.Variable((self.N, 1))
        
        # 在预测中加入扰动项
        disturbance_offset = self.Sw @ w_est_safe.reshape(-1, 1)
        X_pred = self.Sx @ x0 + self.Su @ U + self.total_const_offset + disturbance_offset
        
        # 6. 定义 Cost
        x1_pred = X_pred[0::4, :]
        x2_pred = X_pred[1::4, :]
        gap = x2_pred - (x1_pred + self.mass_width)
        collision_penalty = self.collision_penalty_weight * cp.sum(cp.square(cp.pos(-gap)))
        
        J = cp.quad_form(X_pred - x_ref, self.Q_bar) + cp.quad_form(U, self.R_bar) + collision_penalty
        
        # 7. 【核心修改】定义转换后的约束
        # 我们希望最终的 u_final = u_mpc - w_est_safe 满足 [u_min, u_max]
        # 因此，u_mpc 的约束范围需要平移 + w_est_safe
        # 假设扰动在预测时域内保持常数（与 disturbance_offset 的假设一致）
        # 注意：w_est_safe[0] 是作用在 m1 上的扰动估计，也是前馈补偿的来源
        w1_est = w_est_safe[0]
        
        mpc_u_lower_bound = self.u_min + w1_est
        mpc_u_upper_bound = self.u_max + w1_est
        
        constraints = [
            U <= mpc_u_upper_bound,
            U >= mpc_u_lower_bound
        ]
        
        # 8. 求解
        problem = cp.Problem(cp.Minimize(J), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == 'optimal':
            u_mpc = U.value[0, 0]
        else:
            u_mpc = 0.0  # 失败保底
            
        # 9. 计算最终控制力
        # u_final = MPC 计算出的最优值 - 前馈扰动补偿
        u_final = u_mpc - w1_est
        
        # 10. 【最终安全剪切】
        # 虽然 MPC 内部考虑了约束，但为了防止数值误差或 ESO 估计跳变，
        # 在输出到物理系统前，必须强制限制在物理硬件范围内。
        u_final = np.clip(u_final, self.u_min, self.u_max)
        
        # 记录控制量供下一轮 ESO 使用
        self.last_u = u_final
        # print(f"Time: {time:.2f}s, w_est: {w1_est:.2f}, u_mpc: {u_mpc:.2f}, u_final: {u_final:.2f}")
        return float(u_final)

