clear;
close all;
clc;

%% 1. 参数设置
c = 3e8; % 光速
Fs = 7.14e6; % 采样频率
T_chirp = 98e-6; % Chirp周期
Bw = 3000e6; % 雷达带宽
N = 256; % FFT点数
f_true = 628000; % 固定真实频率
SNR_dB = 5; % 固定信噪比
n_monte = 1000; % Monte Carlo仿真次数
B_fft_res = Fs / N; % FFT的分辨率

% 转换线性信噪比
snr_linear = 10^(SNR_dB / 10);


%% 2. 定义仿真范围与结果存储
% CZT点数变化范围
M_range = 16:8:256; % 从16到256，步长为8
n_m_points = length(M_range);

% 初始化RMSE记录矩阵
rmse_macleod = zeros(n_m_points, 1);
rmse_czt_peak_only = zeros(n_m_points, 1); 
rmse_czt_quad = zeros(n_m_points, 1); 
crlb_freq_theory_rmse = zeros(n_m_points, 1); 

%% 3. 蒙特卡洛仿真主循环
% 外层循环：遍历不同的CZT点数
for i = 1:n_m_points
    current_M = M_range(i);
    
    % 计算当前M值下的CRLB（MSE形式）
    crlb_freq_theory_mse = 3 *Fs^2 / (8 * pi^2 * snr_linear * (current_M^3 + N^3));
    
    % 将CRLB（MSE形式）开方，得到RMSE的CRLB
    crlb_freq_theory_rmse(i) = sqrt(crlb_freq_theory_mse); 
    
    % 初始化当前M值下的临时误差记录数组 (记录平方误差)
    temp_mse_macleod = zeros(n_monte, 1);
    temp_mse_czt_peak_only = zeros(n_monte, 1);
    temp_mse_czt_quad = zeros(n_monte, 1);
    
    % 内层循环：在当前M值下重复n_monte次仿真
    for monte = 1:n_monte
        % 生成含噪声信号
        t = (0:N-1)' / Fs;
        s = exp(1j * 2 * pi * f_true * t);
        noise = (randn(size(t)) + 1j * randn(size(t))) / sqrt(2);
        s_noisy = s * sqrt(snr_linear) + noise;

        %% Step 1: Macleod算法
        X_fft = fft(s_noisy);
        [~, k_fft_peak] = max(abs(X_fft));
        f_fft_peak = (k_fft_peak - 1) * Fs / N;
        [f_macleod, ~, ~] = macleod_algorithm(s_noisy, Fs, N);
        
        %% Step 2-1: FFT-CZT
        f_start = f_fft_peak - B_fft_res / 2;
        f_step = B_fft_res / current_M; % 步长随M变化
        f_axis = f_start + (0:current_M-1) * f_step; % <--- 定义频点
        w = exp(-1j * 2 * pi * f_step / Fs);
        a = exp(1j * 2 * pi * f_start / Fs);
        X_czt = czt(s_noisy, current_M, w, a);
        [~, k_czt_peak_fft] = max(abs(X_czt)); % 用 FFT 峰值定位 CZT
        
        k_czt_peak_fft = max(2, min(k_czt_peak_fft, length(X_czt)-1));
        
        f_czt_peak_only = f_axis(k_czt_peak_fft);
        
        %% Step 2-2: Macleod-CZT
        f_start_macleod = f_macleod - B_fft_res / 2;
        f_step_macleod = B_fft_res / current_M; % 步长随M变化
        f_axis_macleod = f_start_macleod + (0:current_M-1) * f_step_macleod; % <--- 定义频点
        w_macleod = exp(-1j * 2 * pi * f_step_macleod / Fs);
        a_macleod = exp(1j * 2 * pi * f_start_macleod / Fs);
        X_czt_macleod = czt(s_noisy, current_M, w_macleod, a_macleod);
        [~, k_czt_peak_macleod] = max(abs(X_czt_macleod)); % 用 Macleod 结果定位 CZT
        
        k_czt_peak_macleod = max(2, min(k_czt_peak_macleod, length(X_czt_macleod)-1));
        
        
        %% Step 3: CZT二次插值 (基于 Macleod 定位)
        mag_km1 = abs(X_czt_macleod(k_czt_peak_macleod - 1));
        mag_k0  = abs(X_czt_macleod(k_czt_peak_macleod));
        mag_kp1 = abs(X_czt_macleod(k_czt_peak_macleod + 1));
        denom = mag_km1 - 2 * mag_k0 + mag_kp1;
        delta_czt_quad = 0;
        if denom ~= 0
            delta_czt_quad = 0.5 * (mag_km1 - mag_kp1) / denom;
        end
        f_czt_quad = f_axis_macleod(k_czt_peak_macleod) + delta_czt_quad * f_step_macleod;
        
        %% 记录当前迭代的频率平方误差 (MSE)
        % FFT-CZT (纯CZT)
        temp_mse_czt_peak_only(monte) = (f_czt_peak_only - f_true)^2;
        % Macleod-CZT二次插值 (MCZT)
        temp_mse_czt_quad(monte) = (f_czt_quad - f_true)^2;
    end
    
    % 计算当前M值下的平均MSE，并开方得到RMSE，存入主矩阵
    rmse_czt_peak_only(i) = sqrt(mean(temp_mse_czt_peak_only)); 
    rmse_czt_quad(i) = sqrt(mean(temp_mse_czt_quad)); 
end

%% 4. 绘制结果
% 创建图窗
figure;
hold on;
% 绘制CZT峰值曲线
semilogy(M_range, rmse_czt_peak_only, 'g-s', 'DisplayName', 'FFT-CZT', 'LineWidth', 2); 
% 绘制CZT二次插值曲线 (Macleod-CZT)
semilogy(M_range, rmse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT', 'LineWidth', 2); 
% 绘制 CRLB 曲线
semilogy(M_range, crlb_freq_theory_rmse, 'm--', 'DisplayName', 'CRLB', 'LineWidth', 2);

xlabel('Number of CZT Points (M)', 'FontSize', 20);
ylabel('RMSE (Hz)', 'FontSize', 20); 
legend('show');
grid on;
box on;
hold off;

%---
% 调整坐标轴样式
ax = gca;
ax.XColor = 'k'; 
ax.YColor = 'k'; 
ax.LineWidth = 1.5; 
ax.FontSize = 20; 

%% Macleod算法函数 (保持不变)
function [f_est, delta, peak_mag] = macleod_algorithm(x, Fs, N)
    X = fft(x);
    X_abs_sq = abs(X).^2;
    [~, k0] = max(X_abs_sq);
    k0 = k0(1);
    
    k0 = max(2, min(k0, N-1));
    
    X_km1 = X_abs_sq(k0-1);
    X_k0 = X_abs_sq(k0);
    X_kp1 = X_abs_sq(k0+1);
    
    denom = X_km1 - 2*X_k0 + X_kp1;
    delta = 0;
    if denom ~= 0
        delta = (X_km1 - X_kp1)/(2*denom);
    end
    f_est = (k0 - 1 + delta)*Fs/N;
    peak_mag = abs(X(k0));
end