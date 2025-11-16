function main_plotter()
    % ====== Starting Calibration Scheme Comparison Simulation ======
    
    % --- 0. 清理工作区和设置 ---
    clc;
    close all;
    % 抑制在 lstsq 中可能出现的近奇异矩阵警告
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:rankDeficient');
    
    % --- 1. 仿真参数设置 ---
    K = 3;       % K: 发射 (Tx) 天线数量
    L = 4;       % L: 接收 (Rx) 天线数量
    M = K * L;   % M: 虚拟阵元总数
    
    % ILS 迭代参数
    ILS_ITERATIONS = 15;
    ILS_TOLERANCE = 1e-6;
    
    % 蒙特卡洛仿真次数
    N_MONTE_CARLO = 50;           % 为使曲线平滑，每个数据点运行200次取平均
    rng(0);                        % 固定随机种子
    
    fprintf("====== Starting Calibration Scheme Comparison Simulation ======\n");
    fprintf("Fixed parameters: K=%d, L=%d, M=%d, Monte Carlo Runs=%d\n", K, L, M, N_MONTE_CARLO);

    % 固定的仿真参数
    I_FIXED = 6;       % (I < M-1 = 11)
    SNR_FIXED1 = 20;     % 20 dB
    SNR_FIXED2 = 15;     % 15 dB
    ERROR_FIXED = 0.2;   % 真实误差的标准差

    % --- 8. 主函数逻辑 (从 Python 的 if __name__ == "__main__" 移入) ---
    
    % 运行三个绘图函数
    % plot_vs_snr(K, L, I_FIXED, ERROR_FIXED, N_MONTE_CARLO);
    plot_vs_observations(K, L, SNR_FIXED1, ERROR_FIXED, N_MONTE_CARLO);
    % plot_vs_error_magnitude(K, L, SNR_FIXED2, I_FIXED, N_MONTE_CARLO);

    fprintf("\n====== Simulation Finished ======\n");
    fprintf("All plots (.png) and data files (.csv) have been saved to the working directory.\n");
end

% --- 2. 物理参数 ---
% (这些在 MATLAB 中被视为常量，辅助函数可以直接访问它们)
% 在严格的函数式编程中，会把它们作为参数传递，但为了简化转换，
% 我们依赖 MATLAB 的 local function 可以访问主函数工作区变量的特性。
% (注意：这在 plot_vs_snr 等函数中不起作用，因此我们必须传递它们)

% --- 7. 绘图函数 ---

function plot_vs_snr(K, L, I_fixed, error_fixed, N_MONTE_CARLO)

    fprintf("\n--- Generating Plot 1 (vs. SNR) ---\n");
    fprintf("Fixed parameters: I = %d, Error StdDev = %.2f\n", I_fixed, error_fixed);
    
    snr_range = linspace(0, 25, 10); % 0 to 25 dB
    % (VA, MIMO, ILS, CRB_VA, CRB_MIMO)
    results = zeros(length(snr_range), 5);
    
    start_time = tic;
    for i = 1:length(snr_range)
        snr = snr_range(i);
        mse_mc = zeros(N_MONTE_CARLO, 3);
        for mc = 1:N_MONTE_CARLO
            [mse_mc(mc, 1), mse_mc(mc, 2), mse_mc(mc, 3)] = ...
                run_single_simulation(K, L, I_fixed, snr, error_fixed);
        end
        results(i, 1:3) = mean(mse_mc, 1); % 沿第1维求平均
        
        [crb_va, crb_mimo] = calculate_crb(K, L, I_fixed, snr);
        results(i, 4) = crb_va;
        results(i, 5) = crb_mimo;
        
        fprintf("   SNR = %.1f dB complete (%d/%d)\n", snr, i, length(snr_range));
    end
    total_time = toc(start_time);
    
    figure;
    hold on;
    plot(snr_range, results(:, 1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)');
    plot(snr_range, results(:, 2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
    plot(snr_range, results(:, 3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
    plot(snr_range, results(:, 4), 'k--', 'DisplayName', 'CRB (VA)');
    plot(snr_range, results(:, 5), 'k-', 'DisplayName', 'CRB (MIMO)');
    hold off;
    
    xlabel('Signal-to-Noise Ratio (SNR) / dB');
    ylabel('Channel Error MSE (log scale)');
    title(sprintf('Calibration Performance vs. SNR (I=%d, K=%d, L=%d)', I_fixed, K, L));
    set(gca, 'YScale', 'log');
    legend('show');
    grid on;
    ylim([1e-5, inf]);
    saveas(gcf, 'mse_vs_snr.png');
    fprintf("Plot 1 saved to: mse_vs_snr.png\n");
    
    % --- 保存 CSV ---
    csv_filename = 'results_vs_snr.csv';
    df_data = [snr_range(:), results];
    col_names = {'SNR_dB', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'};
    T = array2table(df_data, 'VariableNames', col_names);
    writetable(T, csv_filename);
    fprintf("Data for Plot 1 saved to: %s (Time: %.2fs)\n", csv_filename, total_time);
end

function plot_vs_observations(K, L, snr_fixed, error_fixed, N_MONTE_CARLO)

    fprintf("\n--- Generating Plot 2 (vs. Observations I) ---\n");
    fprintf("Fixed parameters: SNR = %.1f dB, Error StdDev = %.2f\n", snr_fixed, error_fixed);
    M = K * L;
    i_range = max(K,L):5:34; % 从 max(K,L) 到 34 (Python np.arange stop is exclusive)
    results = zeros(length(i_range), 5);
    
    start_time = tic;
    for i = 1:length(i_range)
        obs_I = i_range(i);
        mse_mc = zeros(N_MONTE_CARLO, 3);
        for mc = 1:N_MONTE_CARLO
            [mse_mc(mc, 1), mse_mc(mc, 2), mse_mc(mc, 3)] = ...
                run_single_simulation(K, L, obs_I, snr_fixed, error_fixed);
        end
        results(i, 1:3) = mean(mse_mc, 1);
        
        [crb_va, crb_mimo] = calculate_crb(K, L, obs_I, snr_fixed);
        results(i, 4) = crb_va;
        results(i, 5) = crb_mimo;
        
        fprintf("   I = %d complete (%d/%d)\n", obs_I, i, length(i_range));
    end
    total_time = toc(start_time);
    
    figure;
    hold on;
    plot(i_range, results(:, 1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)');
    plot(i_range, results(:, 2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
    plot(i_range, results(:, 3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
    plot(i_range, results(:, 4), 'k--', 'DisplayName', 'CRB (VA)');
    plot(i_range, results(:, 5), 'k-', 'DisplayName', 'CRB (MIMO)');
    
    % 使用 xline 替代 axvline
    xline(M-1, 'r--', 'DisplayName', sprintf('VA Underdetermined (I < %d)', M-1));
    
    hold off;
    
    xlabel('Number of Independent Observations (I)');
    ylabel('Channel Error MSE (log scale)');
    title(sprintf('Calibration Performance vs. Observations (SNR=%.1fdB)', snr_fixed));
    set(gca, 'YScale', 'log');
    legend('show');
    grid on;
    ylim([1e-5, inf]);
    saveas(gcf, 'mse_vs_observations.png');
    fprintf("Plot 2 saved to: mse_vs_observations.png\n");
    
    % --- 保存 CSV ---
    csv_filename = 'results_vs_observations.csv';
    df_data = [i_range(:), results];
    col_names = {'I_Observations', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'};
    T = array2table(df_data, 'VariableNames', col_names);
    writetable(T, csv_filename);
    fprintf("Data for Plot 2 saved to: %s (Time: %.2fs)\n", csv_filename, total_time);
end

function plot_vs_error_magnitude(K, L, snr_fixed, i_fixed, N_MONTE_CARLO)

    fprintf("\n--- Generating Plot 3 (vs. True Error Variance) ---\n");
    fprintf("Fixed parameters: SNR = %.1f dB, I = %d\n", snr_fixed, i_fixed);
    
    variance_range = linspace(0.05, 0.5, 10); % 误差方差 (sigma^2)
    results = zeros(length(variance_range), 5);
    
    [crb_va, crb_mimo] = calculate_crb(K, L, i_fixed, snr_fixed);
    results(:, 4) = crb_va;
    results(:, 5) = crb_mimo;
    
    start_time = tic;
    
    for i = 1:length(variance_range)
        variance = variance_range(i);
        std_dev = sqrt(variance); % 在仿真前计算标准差
        
        mse_mc = zeros(N_MONTE_CARLO, 3);
        for mc = 1:N_MONTE_CARLO
            [mse_mc(mc, 1), mse_mc(mc, 2), mse_mc(mc, 3)] = ...
                run_single_simulation(K, L, i_fixed, snr_fixed, std_dev);
        end
        results(i, 1:3) = mean(mse_mc, 1);
        fprintf("    Error Variance = %.2f (StdDev = %.2f) complete (%d/%d)\n", variance, std_dev, i, length(variance_range));
    end
    total_time = toc(start_time);
    
    figure;
    hold on;
    plot(variance_range, results(:, 1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)');
    plot(variance_range, results(:, 2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
    plot(variance_range, results(:, 3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
    plot(variance_range, results(:, 4), 'k--', 'DisplayName', 'CRB (VA)');
    plot(variance_range, results(:, 5), 'k-', 'DisplayName', 'CRB (MIMO)');
    hold off;
    
    % 使用 TeX 解释器来显示 \sigma
    xlabel('True Channel Error Variance (Error Var, $\sigma_{\gamma}^2$)', 'Interpreter', 'latex');
    ylabel('Channel Error MSE (log scale)');
    title(sprintf('Calibration Performance vs. True Error Magnitude (SNR=%.1fdB, I=%d)', snr_fixed, i_fixed));
    set(gca, 'YScale', 'log');
    legend('show');
    grid on;
    ylim([1e-5, inf]);
    saveas(gcf, 'mse_vs_error_magnitude.png');
    fprintf("Plot 3 saved to: mse_vs_error_magnitude.png\n");
    
    % --- 保存 CSV ---
    csv_filename = 'results_vs_error_magnitude.csv';
    df_data = [variance_range(:), results];
    col_names = {'Error_Variance', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'};
    T = array2table(df_data, 'VariableNames', col_names);
    writetable(T, csv_filename);
    fprintf("Data for Plot 3 saved to: %s (Time: %.2fs)\n", csv_filename, total_time);
end


% --- 6. 核心仿真封装 ---

function [mse_va, mse_mimo, mse_ils] = run_single_simulation(K, L, I, SNR_dB, error_std_dev)

    M = K * L;
    
    % 物理参数
    f_c = 77e9;
    c = 3e8;
    lambda_c = c / f_c;
    d_rx = lambda_c / 2;
    d_tx = L * d_rx;
    
    % 1. 生成“真实” (Ground Truth) 数据
    gamma_tx_true = ones(K, 1, 'like', 1i); % 列向量
    gamma_rx_true = ones(L, 1, 'like', 1i); % 列向量
    
    err_tx_real = error_std_dev * randn(K-1, 1);
    err_tx_imag = error_std_dev * randn(K-1, 1);
    gamma_tx_true(2:end) = 1.0 + err_tx_real + 1i * err_tx_imag;
    
    err_rx_real = error_std_dev * randn(L-1, 1);
    err_rx_imag = error_std_dev * randn(L-1, 1);
    gamma_rx_true(2:end) = 1.0 + err_rx_real + 1i * err_rx_imag;
    
    gamma_va_true = kron(gamma_tx_true, gamma_rx_true); % (M x 1)
    
    phi_true = linspace(-pi/4, pi/4, I); % (1 x I) 行向量
    alpha_true = (randn(1, I) + 1i * randn(1, I)) / sqrt(2); % (1 x I) 行向量
    H_ideal_true = get_full_steering_matrix(phi_true, K, L, d_tx, d_rx, lambda_c); % (M x I)
    
    % 2. 生成“测量”数据
    Kappa_clean = zeros(M, I, 'like', 1i);
    for i = 1:I
        % (M x 1) .* (M x 1)
        Kappa_clean(:, i) = alpha_true(i) .* gamma_va_true .* H_ideal_true(:, i);
    end
    
    signal_power = mean(abs(Kappa_clean).^2, 'all');
    noise_power = signal_power / (10.^(SNR_dB / 10.0));
    noise = (randn(M, I) + 1i * randn(M, I)) * sqrt(noise_power / 2);
    Kappa_noisy = Kappa_clean + noise; % (M x I)
    
    % 3. 运行三种校准方案
    gamma_va_est = calibrate_va(Kappa_noisy, H_ideal_true); % (M x 1)
    
    [gamma_tx_mimo, gamma_rx_mimo] = calibrate_mimo(Kappa_noisy, phi_true, K, L, d_tx, d_rx, lambda_c); % (K x 1), (L x 1)
    gamma_va_mimo_est = kron(gamma_tx_mimo, gamma_rx_mimo); % (M x 1)
    
    [gamma_tx_ils, gamma_rx_ils] = calibrate_ils(Kappa_noisy, K, L, d_tx, d_rx, lambda_c);
    gamma_va_ils_est = kron(gamma_tx_ils, gamma_rx_ils); % (M x 1)
    
    % 4. 评估结果 (MSE) - 归一化到第一个元素
    mse_va   = mean(abs(gamma_va_est / gamma_va_est(1) - gamma_va_true / gamma_va_true(1)).^2);
    mse_mimo = mean(abs(gamma_va_mimo_est / gamma_va_mimo_est(1) - gamma_va_true / gamma_va_true(1)).^2);
    mse_ils  = mean(abs(gamma_va_ils_est / gamma_va_ils_est(1) - gamma_va_true / gamma_va_true(1)).^2);
end


% --- 5. 核心校准算法 ---

function [gamma_va_est] = calibrate_va(Kappa_noisy, H_ideal_known)

    [M, I] = size(H_ideal_known);
    
    % 逐元素除法
    P_va = Kappa_noisy(2:end, :) ./ Kappa_noisy(1, :);
    H_va = H_ideal_known(2:end, :) ./ H_ideal_known(1, :);
    
    gamma_va_est_partial = zeros(M-1, 1, 'like', 1i);
    
    for m = 1:(M-1)
        p_m_vec = P_va(m, :);
        h_m_vec = H_va(m, :);
        
        % 转换为列向量
        A = h_m_vec(:);
        b = p_m_vec(:);
        
        % Python: gamma_m = np.sum(A.conj()*b) / (np.sum(A.conj()*A) + 1e-6)
        % 注意: A.conj() * b 是逐元素乘法
        gamma_m = sum(conj(A) .* b) / (sum(conj(A) .* A) + 1e-6);
        
        gamma_va_est_partial(m) = gamma_m;
    end
    
    gamma_va_est = [1.0; gamma_va_est_partial];
end

function [gamma_tx_est, gamma_rx_est] = calibrate_mimo(Kappa_noisy, phi_known, K, L, d_tx, d_rx, lambda_c)

    I = length(phi_known);
    
    % MATLAB 默认按列优先重塑，这与 Python (K, L, I) 顺序匹配 M = K*L
    Kappa_reshaped = reshape(Kappa_noisy, K, L, I);
    
    % --- Tx 校准 ---
    gamma_tx_est = ones(K, 1, 'like', 1i);
    for k = 2:K % Python: range(1, K)
        A_tx = [];
        b_tx = [];
        for i = 1:I
            phi = phi_known(i);
            % 索引 k 对应 Python 的 k-1
            h_ki = get_virtual_steering_vector(phi, k-1, 0, d_tx, d_rx, lambda_c) / ...
                   get_virtual_steering_vector(phi, 0, 0, d_tx, d_rx, lambda_c);
            for l = 1:L
                % 索引 k, l, i 对应 Python 的 k-1, l-1, i-1
                % Kappa_reshaped 索引 k, l, i
                % Kappa_reshaped(1, l, i) 对应 Python [0, l-1, i-1]
                p_kli = Kappa_reshaped(k, l, i) / Kappa_reshaped(1, l, i);
                A_tx = [A_tx; h_ki];
                b_tx = [b_tx; p_kli];
            end
        end
        
        gamma_k = A_tx \ b_tx; % 使用 mldivide (lstsq)
        if ~isempty(gamma_k)
            gamma_tx_est(k) = gamma_k(1);
        else
            gamma_tx_est(k) = 1.0;
        end
    end
    
    % --- Rx 校准 ---
    gamma_rx_est = ones(L, 1, 'like', 1i);
    for l = 2:L % Python: range(1, L)
        A_rx = [];
        b_rx = [];
        for i = 1:I
            phi = phi_known(i);
            % 索引 l 对应 Python 的 l-1
            h_li = get_virtual_steering_vector(phi, 0, l-1, d_tx, d_rx, lambda_c) / ...
                   get_virtual_steering_vector(phi, 0, 0, d_tx, d_rx, lambda_c);
            for k = 1:K
                p_kli = Kappa_reshaped(k, l, i) / Kappa_reshaped(k, 1, i);
                A_rx = [A_rx; h_li];
                b_rx = [b_rx; p_kli];
            end
        end
        
        gamma_l = A_rx \ b_rx;
        if ~isempty(gamma_l)
            gamma_rx_est(l) = gamma_l(1);
        else
            gamma_rx_est(l) = 1.0;
        end
    end
end

function [gamma_tx_ils, gamma_rx_ils] = calibrate_ils(Kappa_noisy, K, L, d_tx, d_rx, lambda_c)

    [M, I] = size(Kappa_noisy);
    
    ILS_ITERATIONS = 15;
    ILS_TOLERANCE = 1e-6;
    
    gamma_tx_ils = ones(K, 1, 'like', 1i);
    gamma_rx_ils = ones(L, 1, 'like', 1i);
    phi_ils = zeros(I, 1);
    
    % 初始化角度估计
    for i = 1:I
        phi_ils(i) = simple_doa_estimator(Kappa_noisy(:, i), M, K, L, d_tx, d_rx, lambda_c);
    end
    
    for iter_n = 1:ILS_ITERATIONS
        [gamma_tx_new, gamma_rx_new] = calibrate_mimo(Kappa_noisy, phi_ils, K, L, d_tx, d_rx, lambda_c);
        
        gamma_va_ils = kron(gamma_tx_new, gamma_rx_new);
        % 逐元素除法 (./)，gamma_va_ils(:) 确保广播
        Kappa_calibrated = Kappa_noisy ./ gamma_va_ils(:); 
        
        phi_new = zeros(I, 1);
        for i = 1:I
            phi_new(i) = simple_doa_estimator(Kappa_calibrated(:, i), M, K, L, d_tx, d_rx, lambda_c);
        end
        
        error_tx = norm(gamma_tx_new - gamma_tx_ils) / norm(gamma_tx_ils);
        error_rx = norm(gamma_rx_new - gamma_rx_ils) / norm(gamma_rx_ils);
        
        gamma_tx_ils = gamma_tx_new;
        gamma_rx_ils = gamma_rx_new;
        phi_ils = phi_new;
        
        if (error_tx < ILS_TOLERANCE && error_rx < ILS_TOLERANCE)
            break;
        end
    end
end


% --- 4. CRB 计算函数 ---

function [crb_va_base, crb_mimo] = calculate_crb(K, L, I, SNR_dB)

    snr_linear = 10.^(SNR_dB / 10.0);
    
    % (Eq. 7) CRB_va = (I + sum(SNR_i))^-1
    sum_snr_term = I * snr_linear;
    crb_va_base = 1.0 / (I + sum_snr_term);
    
    % (Eq. 15): CRB_mimo = ((K+L)/(KL)) * CRB_va_base
    crb_mimo = ((K + L) / (K * L)) * crb_va_base;
end


% --- 3. 辅助函数 ---

function [vec_element] = get_virtual_steering_vector(phi, k, l, d_tx, d_rx, lambda_c)
    % 注意: 这里的 k 和 l 遵循 Python 的 0 索引逻辑 (0 到 K-1, 0 到 L-1)

    phase = -2i * pi * (k * d_tx + l * d_rx) * sin(phi) / lambda_c;
    vec_element = exp(phase);
end

function [H_ideal] = get_full_steering_matrix(phi_array, K, L, d_tx, d_rx, lambda_c)

    M = K * L;
    I = length(phi_array);
    H_ideal = zeros(M, I, 'like', 1i);
    
    for i = 1:I
        phi = phi_array(i);
        m = 1; % MATLAB 1-based index
        for k = 1:K
            for l = 1:L
                % 传递 k-1 和 l-1 以匹配 0 索引逻辑
                H_ideal(m, i) = get_virtual_steering_vector(phi, k-1, l-1, d_tx, d_rx, lambda_c);
                m = m + 1;
            end
        end
    end
end

function [est_angle] = simple_doa_estimator(x_calibrated, M, K, L, d_tx, d_rx, lambda_c)

    angle_grid = linspace(-pi/2, pi/2, 361);
    spectrum = zeros(1, length(angle_grid));
    
    for i = 1:length(angle_grid)
        phi = angle_grid(i);
        % get_full_steering_matrix 接收数组，phi 是标量，它会处理
        a_phi = get_full_steering_matrix(phi, K, L, d_tx, d_rx, lambda_c);
        % a_phi(:) 展平为列向量
        % a_phi' * x_calibrated 是 (conj(a_phi) * x_calibrated)
        spectrum(i) = abs(a_phi' * x_calibrated).^2;
    end
    
    [~, peak_index] = max(spectrum);
    est_angle = angle_grid(peak_index);
end