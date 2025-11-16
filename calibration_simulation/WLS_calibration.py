import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# --- [NEW] 辅助函数 1: 标准 ALS 幅度估计 ---
def calibrate_amplitude_als(y, max_iter=20, tol=1e-4):
    """
    使用标准 ALS 迭代估计幅度 (对应您原始文档的思路)
    J = sum( (y_ij - a_tx_i * a_rx_j)^2 )
    """
    K, L = y.shape
    # 初始化
    alpha_tx = np.ones(K)
    alpha_rx = y[0, :] / (np.mean(y[0, :]) + 1e-9) # 使用数据初始化

    for _ in range(max_iter):
        alpha_tx_old = np.copy(alpha_tx)
        alpha_rx_old = np.copy(alpha_rx)

        # 1. 更新 Tx (i=0 to K-1)
        for i in range(K):
            # num = sum(y_ij * a_rx_j)
            # den = sum(a_rx_j^2)
            num = np.dot(y[i, :], alpha_rx)
            den = np.dot(alpha_rx, alpha_rx)
            alpha_tx[i] = num / (den + 1e-9)

        # 2. 更新 Rx (j=0 to L-1)
        for j in range(L):
            # num = sum(y_ij * a_tx_i)
            # den = sum(a_tx_i^2)
            num = np.dot(y[:, j], alpha_tx)
            den = np.dot(alpha_tx, alpha_tx)
            alpha_rx[j] = num / (den + 1e-9)

        if np.linalg.norm(alpha_tx - alpha_tx_old) < tol and \
           np.linalg.norm(alpha_rx - alpha_rx_old) < tol:
            break

    # 归一化：我们必须设定一个参考，例如 a_tx[0] = 1
    # a_tx_norm * a_rx_norm = (a_tx / a_tx_0) * (a_rx * a_tx_0)
    norm_factor = alpha_tx[0]
    alpha_tx_est = alpha_tx / (norm_factor + 1e-9)
    alpha_rx_est = alpha_rx * norm_factor

    return alpha_tx_est, alpha_rx_est

# --- [NEW] 辅助函数 2: 加权 W-ALS 幅度估计 ---
def calibrate_amplitude_w_als(y, w, max_iter=20, tol=1e-4):
    """
    使用加权 W-ALS 迭代估计幅度
    J = sum( w_ij * (y_ij - a_tx_i * a_rx_j)^2 )
    """
    K, L = y.shape
    # 初始化
    alpha_tx = np.ones(K)
    alpha_rx = y[0, :] / (np.mean(y[0, :]) + 1e-9)

    for _ in range(max_iter):
        alpha_tx_old = np.copy(alpha_tx)
        alpha_rx_old = np.copy(alpha_rx)

        # 1. 更新 Tx (i=0 to K-1)
        for i in range(K):
            # num = sum(w_ij * y_ij * a_rx_j)
            # den = sum(w_ij * a_rx_j^2)
            num = np.dot(y[i, :] * w[i, :], alpha_rx)
            den = np.dot(w[i, :], alpha_rx**2)
            alpha_tx[i] = num / (den + 1e-9)

        # 2. 更新 Rx (j=0 to L-1)
        for j in range(L):
            # num = sum(w_ij * y_ij * a_tx_i)
            # den = sum(w_ij * a_tx_i^2)
            num = np.dot(y[:, j] * w[:, j], alpha_tx)
            den = np.dot(w[:, j], alpha_tx**2)
            alpha_rx[j] = num / (den + 1e-9)

        if np.linalg.norm(alpha_tx - alpha_tx_old) < tol and \
           np.linalg.norm(alpha_rx - alpha_rx_old) < tol:
            break

    # 归一化
    norm_factor = alpha_tx[0]
    alpha_tx_est = alpha_tx / (norm_factor + 1e-9)
    alpha_rx_est = alpha_rx * norm_factor

    return alpha_tx_est, alpha_rx_est


# --- 1. [MODIFIED] 核心算法：标准 LS/ALS ---
def calibrate_ls(z_obs):
    """
    使用标准 LS/ALS 估计幅度和相位误差。
    """
    K, L = z_obs.shape
    y = np.abs(z_obs)

    # 1a. 幅度校准 (Standard ALS)
    # [MODIFIED] 调用迭代 ALS
    alpha_tx_est, alpha_rx_est = calibrate_amplitude_als(y)

    # 1b. 相位校准 (Standard LS)
    theta = np.angle(z_obs)
    theta_ref = theta[0, 0]
    num_unknowns = (K - 1) + (L - 1)
    num_eqs = (K * L) - 1

    if num_unknowns == 0:
        return np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])

    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)

    eq_idx = 0
    for i in range(K):
        for j in range(L):
            if i == 0 and j == 0:
                continue
            if i > 0:
                A[eq_idx, i - 1] = 1.0
            if j > 0:
                A[eq_idx, (K - 1) + j - 1] = 1.0
            b[eq_idx] = np.angle(np.exp(1j * (theta[i, j] - theta_ref)))
            eq_idx += 1

    try:
        x_phase = np.linalg.pinv(A) @ b
    except np.linalg.LinAlgError:
        x_phase = np.zeros(num_unknowns)

    phi_tx_est = np.concatenate(([0.0], x_phase[0 : K - 1]))
    phi_rx_est = np.concatenate(([0.0], x_phase[K - 1 :]))

    return alpha_tx_est, alpha_rx_est, phi_tx_est, phi_rx_est

# --- 2. [MODIFIED] 核心算法：加权 WLS/W-ALS ---
def calibrate_wls(z_obs, noise_var_matrix_snapshot, n_obs):
    """
    使用加权 W-ALS/WLS 估计幅度和相位误差。
    """
    K, L = z_obs.shape
    y = np.abs(z_obs)

    # 2a. 权重计算
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        noise_var_avg = noise_var_matrix_snapshot / n_obs
        w = (y**2 - noise_var_avg) / noise_var_avg
    w = np.maximum(w, 1e-3)
    w = np.nan_to_num(w, nan=1e-3, posinf=1e5, neginf=1e-3)

    # 2b. 幅度校准 (W-ALS)
    # [MODIFIED] 调用加权迭代 W-ALS
    alpha_tx_est, alpha_rx_est = calibrate_amplitude_w_als(y, w)

    # 2c. 相位校准 (WLS)
    theta = np.angle(z_obs)
    theta_ref = theta[0, 0]
    num_unknowns = (K - 1) + (L - 1)
    num_eqs = (K * L) - 1

    if num_unknowns == 0:
        return np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])

    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)
    W_diag = np.zeros(num_eqs)

    eq_idx = 0
    for i in range(K):
        for j in range(L):
            if i == 0 and j == 0:
                continue

            if i > 0:
                A[eq_idx, i - 1] = 1.0
            if j > 0:
                A[eq_idx, (K - 1) + j - 1] = 1.0
            b[eq_idx] = np.angle(np.exp(1j * (theta[i, j] - theta_ref)))
            W_diag[eq_idx] = w[i, j]
            eq_idx += 1

    W = np.diag(W_diag)

    try:
        pinv_matrix = np.linalg.inv(A.T @ W @ A) @ A.T @ W
        x_phase = pinv_matrix @ b
    except np.linalg.LinAlgError:
        try:
            x_phase = np.linalg.pinv(A) @ b
        except np.linalg.LinAlgError:
            x_phase = np.zeros(num_unknowns)

    phi_tx_est = np.concatenate(([0.0], x_phase[0 : K - 1]))
    phi_rx_est = np.concatenate(([0.0], x_phase[K - 1 :]))

    return alpha_tx_est, alpha_rx_est, phi_tx_est, phi_rx_est


# --- 3. 仿真数据生成器 ---
# (与之前版本相同，保持不变)
def generate_data(K, L, true_amp_tx, true_amp_rx, true_phase_tx, true_phase_rx,
                  snr_db, n_obs, noise_std_ratio_per_rx):
    s = 1.0 + 0.0j
    z_ideal = np.zeros((K, L), dtype=complex)
    for i in range(K):
        for j in range(L):
            gamma_tx = true_amp_tx[i] * np.exp(1j * true_phase_tx[i])
            gamma_rx = true_amp_rx[j] * np.exp(1j * true_phase_rx[j])
            z_ideal[i, j] = gamma_tx * gamma_rx * s

    signal_power_ref = np.abs(z_ideal[0, 0])**2
    snr_linear = 10**(snr_db / 10.0)
    noise_var_snapshot_ref = signal_power_ref / snr_linear
    noise_std_snapshot_ref = np.sqrt(noise_var_snapshot_ref / 2.0)

    noise_std_matrix = np.zeros((K, L))
    noise_var_matrix_snapshot = np.zeros((K, L))

    for j in range(L):
        ratio = noise_std_ratio_per_rx[j]
        std_j = noise_std_snapshot_ref * ratio
        var_j = std_j**2 * 2
        noise_std_matrix[:, j] = std_j
        noise_var_matrix_snapshot[:, j] = var_j

    z_obs_total = np.zeros((K, L), dtype=complex)
    for _ in range(n_obs):
        noise_i = np.random.normal(0, 1.0, (K, L)) * noise_std_matrix
        noise_q = np.random.normal(0, 1.0, (K, L)) * noise_std_matrix
        noise = noise_i + 1j * noise_q
        z_obs_total += (z_ideal + noise)

    z_obs_avg = z_obs_total / n_obs

    return z_obs_avg, noise_var_matrix_snapshot


# --- 4. CRB 计算函数 ---
# (与之前版本相同，保持不变)
def calculate_crb(K, L, snr_db_snapshot, n_obs):
    if K <= 0 or L <= 0 or (K == 1 and L == 1):
        return 1e3, 1e3

    snr_linear_snapshot = 10**(snr_db_snapshot / 10.0)
    snr_linear_avg = n_obs * snr_linear_snapshot
    denominator = snr_linear_avg + 1.0
    if denominator < 1e-9:
        return 1e3, 1e3

    crb_tx_param = 1.0 / (L * denominator)
    crb_rx_param = 1.0 / (K * denominator)

    total_se_crb = (K - 1.0) * crb_tx_param + (L - 1.0) * crb_rx_param
    crb_avg_mse = total_se_crb / (K + L)

    return crb_avg_mse, crb_avg_mse

# --- 5. 绘图与保存函数 ---
# (与之前版本相同，保持不变)
def plot_and_save_results(x_data, mse_results, x_label, title):
    try:
        data_to_save = {
            x_label: x_data,
            'MSE_LS_Amp': mse_results['ls_amp'],
            'MSE_WLS_Amp': mse_results['wls_amp'],
            'CRB_Amp': mse_results['crb_amp'],
            'MSE_LS_Phase': mse_results['ls_phase'],
            'MSE_WLS_Phase': mse_results['wls_phase'],
            'CRB_Phase': mse_results['crb_phase']
        }
        df = pd.DataFrame(data_to_save)

        csv_filename = title.lower().replace(" ", "_").replace(":", "").replace(".", "") + ".csv"
        df.to_csv(csv_filename, index=False)
        print(f"Plot data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    # 幅度 MSE
    ax1.plot(x_data, mse_results['ls_amp'], 'o-', label='Standard LS (ALS)')
    ax1.plot(x_data, mse_results['wls_amp'], 's-', label='WLS (W-ALS)')
    ax1.plot(x_data, mse_results['crb_amp'], 'k--', label='CRB')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Amplitude MSE (log scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    # 相位 MSE
    ax2.plot(x_data, mse_results['ls_phase'], 'o-', label='Standard LS')
    ax2.plot(x_data, mse_results['wls_phase'], 's-', label='WLS')
    ax2.plot(x_data, mse_results['crb_phase'], 'k--', label='CRB')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Phase MSE (rad^2) (log scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    png_filename = title.lower().replace(" ", "_").replace(":", "").replace(".", "") + ".png"
    plt.savefig(png_filename)
    print(f"Plot image successfully saved to {png_filename}")
    # plt.show()

# --- 6. 仿真主循环 ---
# (与之前版本相同，保持不变)
def run_simulation_mse(K, L, n_monte_carlo, default_snr, default_err_amp, default_err_phase, default_n_obs,
                       noise_std_ratio_per_rx):

    warnings.filterwarnings('ignore')

    # --- 实验一: MSE vs. SNR ---
    print(f"\n--- Running Experiment 1: MSE vs. SNR (Noise Ratio: {noise_std_ratio_per_rx}) ---")
    snr_db_range = np.linspace(0, 40, 21)
    mse_results_snr = {k: [] for k in ['ls_amp', 'ls_phase', 'wls_amp', 'wls_phase', 'crb_amp', 'crb_phase']}

    for snr_db in snr_db_range:
        se_ls_amp, se_ls_phase, se_wls_amp, se_wls_phase = 0, 0, 0, 0
        crb_amp, crb_phase = calculate_crb(K, L, snr_db, default_n_obs)

        true_amp_tx = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(K-1)])
        true_amp_rx = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(L-1)])
        true_phase_tx = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(K-1)])
        true_phase_rx = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(L-1)])

        for _ in range(n_monte_carlo):
            z_obs, noise_var_matrix = generate_data(K, L, true_amp_tx, true_amp_rx, true_phase_tx, true_phase_rx,
                                                    snr_db, default_n_obs, noise_std_ratio_per_rx)

            ls_a_tx, ls_a_rx, ls_p_tx, ls_p_rx = calibrate_ls(z_obs)
            wls_a_tx, wls_a_rx, wls_p_tx, wls_p_rx = calibrate_wls(z_obs, noise_var_matrix, default_n_obs)

            se_ls_amp += np.sum((ls_a_tx - true_amp_tx)**2) + np.sum((ls_a_rx - true_amp_rx)**2)
            se_wls_amp += np.sum((wls_a_tx - true_amp_tx)**2) + np.sum((wls_a_rx - true_amp_rx)**2)
            p_err_ls = np.angle(np.exp(1j * (np.concatenate((ls_p_tx, ls_p_rx)) - np.concatenate((true_phase_tx, true_phase_rx)))))
            p_err_wls = np.angle(np.exp(1j * (np.concatenate((wls_p_tx, wls_p_rx)) - np.concatenate((true_phase_tx, true_phase_rx)))))
            se_ls_phase += np.sum(p_err_ls**2)
            se_wls_phase += np.sum(p_err_wls**2)

        mse_results_snr['ls_amp'].append(se_ls_amp / (n_monte_carlo * (K+L)))
        mse_results_snr['ls_phase'].append(se_ls_phase / (n_monte_carlo * (K+L)))
        mse_results_snr['wls_amp'].append(se_wls_amp / (n_monte_carlo * (K+L)))
        mse_results_snr['wls_phase'].append(se_wls_phase / (n_monte_carlo * (K+L)))
        mse_results_snr['crb_amp'].append(crb_amp)
        mse_results_snr['crb_phase'].append(crb_phase)

    plot_and_save_results(snr_db_range, mse_results_snr, "SNR (dB)", "Experiment 1: MSE vs. SNR")

    # --- 实验二: MSE vs. 误差大小 ---
    print(f"\n--- Running Experiment 2: MSE vs. Error Magnitude (Noise Ratio: {noise_std_ratio_per_rx}) ---")
    err_phase_range_deg = np.linspace(0, 45, 16)
    err_phase_range_rad = np.deg2rad(err_phase_range_deg)
    mse_results_err = {k: [] for k in ['ls_amp', 'ls_phase', 'wls_amp', 'wls_phase', 'crb_amp', 'crb_phase']}

    crb_amp, crb_phase = calculate_crb(K, L, default_snr, default_n_obs)

    for err_phase in err_phase_range_rad:
        err_amp = err_phase / (np.pi/2) * 0.5
        se_ls_amp, se_ls_phase, se_wls_amp, se_wls_phase = 0, 0, 0, 0

        for _ in range(n_monte_carlo):
            true_amp_tx = np.array([1.0] + [np.random.normal(1.0, err_amp) for _ in range(K-1)])
            true_amp_rx = np.array([1.0] + [np.random.normal(1.0, err_amp) for _ in range(L-1)])
            true_phase_tx = np.array([0.0] + [np.random.normal(0.0, err_phase) for _ in range(K-1)])
            true_phase_rx = np.array([0.0] + [np.random.normal(0.0, err_phase) for _ in range(L-1)])

            z_obs, noise_var_matrix = generate_data(K, L, true_amp_tx, true_amp_rx, true_phase_tx, true_phase_rx,
                                                    default_snr, default_n_obs, noise_std_ratio_per_rx)

            ls_a_tx, ls_a_rx, ls_p_tx, ls_p_rx = calibrate_ls(z_obs)
            wls_a_tx, wls_a_rx, wls_p_tx, wls_p_rx = calibrate_wls(z_obs, noise_var_matrix, default_n_obs)

            se_ls_amp += np.sum((ls_a_tx - true_amp_tx)**2) + np.sum((ls_a_rx - true_amp_rx)**2)
            se_wls_amp += np.sum((wls_a_tx - true_amp_tx)**2) + np.sum((wls_a_rx - true_amp_rx)**2)
            p_err_ls = np.angle(np.exp(1j * (np.concatenate((ls_p_tx, ls_p_rx)) - np.concatenate((true_phase_tx, true_phase_rx)))))
            p_err_wls = np.angle(np.exp(1j * (np.concatenate((wls_p_tx, wls_p_rx)) - np.concatenate((true_phase_tx, true_phase_rx)))))
            se_ls_phase += np.sum(p_err_ls**2)
            se_wls_phase += np.sum(p_err_wls**2)

        mse_results_err['ls_amp'].append(se_ls_amp / (n_monte_carlo * (K+L)))
        mse_results_err['ls_phase'].append(se_ls_phase / (n_monte_carlo * (K+L)))
        mse_results_err['wls_amp'].append(se_wls_amp / (n_monte_carlo * (K+L)))
        mse_results_err['wls_phase'].append(se_wls_phase / (n_monte_carlo * (K+L)))
        mse_results_err['crb_amp'].append(crb_amp)
        mse_results_err['crb_phase'].append(crb_phase)

    plot_and_save_results(err_phase_range_deg, mse_results_err, "Error Magnitude (Phase StDev, deg)", "Experiment 2: MSE vs. Error Magnitude")

    # --- 实验三: MSE vs. 观测次数 (Chirp 累计) ---
    print(f"\n--- Running Experiment 3: MSE vs. Number of Observations (Noise Ratio: {noise_std_ratio_per_rx}) ---")
    n_obs_range = np.arange(1, 21)
    mse_results_obs = {k: [] for k in ['ls_amp', 'ls_phase', 'wls_amp', 'wls_phase', 'crb_amp', 'crb_phase']}

    true_amp_tx_fixed = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(K-1)])
    true_amp_rx_fixed = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(L-1)])
    true_phase_tx_fixed = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(K-1)])
    true_phase_rx_fixed = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(L-1)])

    for n_obs in n_obs_range:
        se_ls_amp, se_ls_phase, se_wls_amp, se_wls_phase = 0, 0, 0, 0
        crb_amp, crb_phase = calculate_crb(K, L, default_snr, n_obs)

        for _ in range(n_monte_carlo):
            z_obs, noise_var_matrix = generate_data(K, L, true_amp_tx_fixed, true_amp_rx_fixed, true_phase_tx_fixed, true_phase_rx_fixed,
                                                    default_snr, n_obs, noise_std_ratio_per_rx)

            ls_a_tx, ls_a_rx, ls_p_tx, ls_p_rx = calibrate_ls(z_obs)
            wls_a_tx, wls_a_rx, wls_p_tx, wls_p_rx = calibrate_wls(z_obs, noise_var_matrix, n_obs)

            se_ls_amp += np.sum((ls_a_tx - true_amp_tx_fixed)**2) + np.sum((ls_a_rx - true_amp_rx_fixed)**2)
            se_wls_amp += np.sum((wls_a_tx - true_amp_tx_fixed)**2) + np.sum((wls_a_rx - true_amp_rx_fixed)**2)
            p_err_ls = np.angle(np.exp(1j * (np.concatenate((ls_p_tx, ls_p_rx)) - np.concatenate((true_phase_tx_fixed, true_phase_rx_fixed)))))
            p_err_wls = np.angle(np.exp(1j * (np.concatenate((wls_p_tx, wls_p_rx)) - np.concatenate((true_phase_tx_fixed, true_phase_rx_fixed)))))
            se_ls_phase += np.sum(p_err_ls**2)
            se_wls_phase += np.sum(p_err_wls**2)

        mse_results_obs['ls_amp'].append(se_ls_amp / (n_monte_carlo * (K+L)))
        mse_results_obs['ls_phase'].append(se_ls_phase / (n_monte_carlo * (K+L)))
        mse_results_obs['wls_amp'].append(se_wls_amp / (n_monte_carlo * (K+L)))
        mse_results_obs['wls_phase'].append(se_wls_phase / (n_monte_carlo * (K+L)))
        mse_results_obs['crb_amp'].append(crb_amp)
        mse_results_obs['crb_phase'].append(crb_phase)

    plot_and_save_results(n_obs_range, mse_results_obs, "Number of Observations (Chirps)", "Experiment 3: MSE vs. N_obs")


# --- 实验四: 波束图对比 ---
# [MODIFIED] 传入异质噪声参数
def run_simulation_beampattern(K, L, default_snr, default_err_amp, default_err_phase, default_n_obs,
                               noise_std_ratio_per_rx):
    print(f"\n--- Running Experiment 4: Beampattern Comparison (Noise Ratio: {noise_std_ratio_per_rx}) ---")

    M = K * L
    d_lambda = 0.5
    tx_pos = np.arange(K) * L
    rx_pos = np.arange(L)
    v_pos = np.array([i+j for i in tx_pos for j in rx_pos]) * d_lambda

    true_amp_tx = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(K-1)])
    true_amp_rx = np.array([1.0] + [np.random.normal(1.0, default_err_amp) for _ in range(L-1)])
    true_phase_tx = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(K-1)])
    true_phase_rx = np.array([0.0] + [np.random.normal(0.0, default_err_phase) for _ in range(L-1)])

    true_gamma = np.zeros(M, dtype=complex)
    k = 0
    for i in range(K):
        for j in range(L):
            true_gamma[k] = (true_amp_tx[i] * np.exp(1j * true_phase_tx[i])) * \
                            (true_amp_rx[j] * np.exp(1j * true_phase_rx[j]))
            k += 1

    z_obs, noise_var_matrix = generate_data(K, L, true_amp_tx, true_amp_rx, true_phase_tx, true_phase_rx,
                                            default_snr, default_n_obs, noise_std_ratio_per_rx)

    ls_a_tx, ls_a_rx, ls_p_tx, ls_p_rx = calibrate_ls(z_obs)
    wls_a_tx, wls_a_rx, wls_p_tx, wls_p_rx = calibrate_wls(z_obs, noise_var_matrix, default_n_obs)

    ls_gamma_est = np.zeros(M, dtype=complex)
    wls_gamma_est = np.zeros(M, dtype=complex)
    k = 0
    for i in range(K):
        for j in range(L):
            ls_gamma_est[k] = (ls_a_tx[i] * np.exp(1j * ls_p_tx[i])) * \
                              (ls_a_rx[j] * np.exp(1j * ls_p_rx[j]))
            wls_gamma_est[k] = (wls_a_tx[i] * np.exp(1j * wls_p_tx[i])) * \
                               (wls_a_rx[j] * np.exp(1j * wls_p_rx[j]))
            k += 1

    angles_deg = np.linspace(-90, 90, 401)
    angles_rad = np.deg2rad(angles_deg)

    steering_vectors = np.exp(1j * 2 * np.pi * np.outer(v_pos, np.sin(angles_rad)))

    gamma_ideal = np.ones(M, dtype=complex)
    bp_ideal = np.abs(gamma_ideal @ steering_vectors)

    gamma_uncal = true_gamma
    bp_uncal = np.abs(gamma_uncal @ steering_vectors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gamma_ls_cal = true_gamma / ls_gamma_est
    gamma_ls_cal = np.nan_to_num(gamma_ls_cal, nan=1.0)
    bp_ls_cal = np.abs(gamma_ls_cal @ steering_vectors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gamma_wls_cal = true_gamma / wls_gamma_est
    gamma_wls_cal = np.nan_to_num(gamma_wls_cal, nan=1.0)
    bp_wls_cal = np.abs(gamma_wls_cal @ steering_vectors)

    bp_ideal_db = 20 * np.log10(bp_ideal / np.max(bp_ideal))
    bp_uncal_db = 20 * np.log10(bp_uncal / np.max(bp_uncal))
    bp_ls_cal_db = 20 * np.log10(bp_ls_cal / np.max(bp_ls_cal))
    bp_wls_cal_db = 20 * np.log10(bp_wls_cal / np.max(bp_wls_cal))

    plt.figure(figsize=(10, 7))
    plt.plot(angles_deg, bp_ideal_db, 'k', label='Ideal (No Error)', linewidth=2)
    plt.plot(angles_deg, bp_uncal_db, 'r:', label='Uncalibrated', linewidth=2)
    plt.plot(angles_deg, bp_ls_cal_db, 'g--', label='LS Calibrated', linewidth=2)
    plt.plot(angles_deg, bp_wls_cal_db, 'b-', label='WLS Calibrated (Proposed)', linewidth=2)

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized Beampattern (dB)")
    plt.title(f"Experiment 4: Beampattern Comparison (SNR={default_snr}dB, N_obs={default_n_obs})")
    plt.legend()
    plt.grid(True)
    plt.ylim([-50, 1])

    try:
        df_beampattern = pd.DataFrame({
            "Angle_deg": angles_deg,
            "Ideal_dB": bp_ideal_db,
            "Uncalibrated_dB": bp_uncal_db,
            "LS_Calibrated_dB": bp_ls_cal_db,
            "WLS_Calibrated_dB": bp_wls_cal_db
        })
        csv_filename = "experiment_4_beampattern_comparison.csv"
        df_beampattern.to_csv(csv_filename, index=False)
        print(f"Plot data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"Error saving beampattern CSV file: {e}")

    png_filename = "experiment_4_beampattern_comparison.png"
    plt.savefig(png_filename)
    print(f"Plot image successfully saved to {png_filename}")
    # plt.show()


# --- 7. 主函数入口 ---
if __name__ == "__main__":

    # --- 基本仿真参数 ---
    K_TX = 2  # 发射天线数
    L_RX = 2  # 接收天线数
    N_MONTE_CARLO = 1000  # Monte-Carlo 仿真次数

    # --- 实验的默认值 ---
    DEFAULT_SNR_DB = 20.0       # 默认信噪比
    DEFAULT_ERR_PHASE = np.deg2rad(20.0) # 默认相位误差标准差 (20度)
    DEFAULT_ERR_AMP = 0.50      # 默认幅度误差标准差 (0.20)
    DEFAULT_N_OBS = 1           # 默认观测次数

    # --- [NEW] 定义异质噪声 ---
    # 假设 RX0 (j=0) 噪声标准差比例为 1.0
    # 假设 RX1 (j=1) 噪声标准差比例为 5.0 (即 25倍 噪声功率)
    NOISE_STD_RATIO_PER_RX = [1.0, 5.0]

    print("Starting simulation...")

    # 运行三个 MSE 实验
    # run_simulation_mse(K_TX, L_RX, N_MONTE_CARLO,
    #                    DEFAULT_SNR_DB, DEFAULT_ERR_AMP, DEFAULT_ERR_PHASE, DEFAULT_N_OBS,
    #                    NOISE_STD_RATIO_PER_RX)

    # 运行波束图对比实验
    run_simulation_beampattern(K_TX, L_RX, DEFAULT_SNR_DB,
                               DEFAULT_ERR_AMP, DEFAULT_ERR_PHASE, DEFAULT_N_OBS,
                               NOISE_STD_RATIO_PER_RX)

    print("\nSimulation finished. All CSV and PNG files saved.")