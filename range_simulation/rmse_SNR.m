clear;
close all;
clc;

%% 1. 参数设置
c = 3e8; % 光速
Fs = 7.14e6; % 采样频率
T_chirp = 98e-6; % Chirp周期
Bw = 3000e6; % 雷达带宽
N = 256; % FFT点数
f_true = 628000; % 真实频率
n_monte = 1000; % Monte Carlo仿真次数
M = 32; % Chirp-Z点数
B_fft_res = Fs / N; % FFT的分辨率


%% 2. 定义仿真范围与结果存储
SNR_dB_range = -5:1:20; % 信噪比（dB）范围
n_snr_points = length(SNR_dB_range);

% 初始化RMSE记录矩阵
rmse_fft_peak = zeros(n_snr_points, 1);
rmse_macleod = zeros(n_snr_points, 1); 
rmse_czt_peak_only = zeros(n_snr_points, 1); 
rmse_czt_quad = zeros(n_snr_points, 1); 
crlb_freq_theory_rmse = zeros(n_snr_points, 1); 

%% 3. 蒙特卡洛仿真主循环
% 外层循环：遍历不同的信噪比
for i = 1:n_snr_points
    current_SNR_dB = SNR_dB_range(i);
    snr_linear = 10^(current_SNR_dB / 10);
    
    % 计算当前SNR下的CRLB理论值 (MSE形式)
    B = 1 * B_fft_res;
    crlb_freq_theory_mse = 3 *Fs^2 / (8 * pi^2 * snr_linear * (M^3 + N^3));
    
    % 将CRLB（MSE形式）开方，得到RMSE的CRLB
    crlb_freq_theory_rmse(i) = sqrt(crlb_freq_theory_mse); 
    
    % 初始化当前SNR下的临时误差记录数组 (记录的是平方误差)
    temp_mse_fft = zeros(n_monte, 1);
    temp_mse_macleod = zeros(n_monte, 1);
    temp_mse_czt_peak_only = zeros(n_monte, 1);
    temp_mse_czt_quad = zeros(n_monte, 1);
    
    % 内层循环：在当前SNR下重复n_monte次仿真
    for monte = 1:n_monte
        % 生成含噪声信号
        t = (0:N-1)' / Fs;
        s = exp(1j * 2 * pi * f_true * t);
        noise = (randn(size(t)) + 1j * randn(size(t))) / sqrt(2);
        s_noisy = s * sqrt(snr_linear) + noise;
        
        %% Step 0: FFT频率估计
        X_fft = fft(s_noisy);
        [~, k_fft_peak] = max(abs(X_fft));
        f_fft_peak = (k_fft_peak - 1) * Fs / N;
        
        %% Step 2: FFT Chirp-Z变换 (CZT) - 第一次CZT用于对比
        % 以FFT峰值点为中心，进行CZT
        f_start = f_fft_peak - B_fft_res / 2;
        f_step = B_fft_res / M;
        f_axis = f_start + (0:M-1) * f_step;
        w = exp(-1j * 2 * pi * f_step / Fs);
        a = exp(1j * 2 * pi * f_start / Fs);
        X_czt = czt(s_noisy, M, w, a);
        [~, k_czt_peak] = max(abs(X_czt));
        
        % 确保索引在有效范围内
        k_czt_peak = max(2, min(k_czt_peak, length(X_czt)-1));
        
        % 仅用峰值位置估计频率
        f_czt_peak_only = f_axis(k_czt_peak);

        %% Step 1: Macleod算法
        [f_macleod, ~, ~] = macleod_algorithm(s_noisy, Fs, N);

        %% Step 2-2: Macleod-CZT (以Macleod结果为中心)
        % 以Macleod算法得到的频率作为CZT的搜索中心
        f_start_macleod = f_macleod - B_fft_res / 2;
        f_step_macleod = B_fft_res / M;
        f_axis_macleod = f_start_macleod + (0:M-1) * f_step_macleod;
        w_macleod = exp(-1j * 2 * pi * f_step_macleod / Fs);
        a_macleod = exp(1j * 2 * pi * f_start_macleod / Fs);
        X_czt_macleod = czt(s_noisy, M, w_macleod, a_macleod);
        [~, k_czt_peak_macleod] = max(abs(X_czt_macleod));
        
        % 确保索引在有效范围内
        k_czt_peak_macleod = max(2, min(k_czt_peak_macleod, length(X_czt_macleod)-1));
        
        
        %% Step 3: CZT二次插值 (Macleod-CZT)
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
        temp_mse_fft(monte) = (f_fft_peak - f_true)^2;
        temp_mse_macleod(monte) = (f_macleod - f_true)^2;
        temp_mse_czt_peak_only(monte) = (f_czt_peak_only - f_true)^2;
        temp_mse_czt_quad(monte) = (f_czt_quad - f_true)^2;
    end
    
    % 计算当前SNR下的平均平方误差(MSE)，并开方得到RMSE，存入主矩阵
    rmse_fft_peak(i) = sqrt(mean(temp_mse_fft)); 
    rmse_macleod(i) = sqrt(mean(temp_mse_macleod)); 
    rmse_czt_peak_only(i) = sqrt(mean(temp_mse_czt_peak_only));
    rmse_czt_quad(i) = sqrt(mean(temp_mse_czt_quad)); 
end

%% 4. 绘制结果
figure;
hold on;

% 使用semilogy绘制半对数曲线，更清晰
semilogy(SNR_dB_range, rmse_fft_peak, 'r-o', 'DisplayName', 'FFT', 'LineWidth', 2); 
semilogy(SNR_dB_range, rmse_macleod, 'b-^', 'DisplayName', 'Macleod', 'LineWidth', 2);
semilogy(SNR_dB_range, rmse_czt_peak_only, 'g-s', 'DisplayName', 'CZT', 'LineWidth', 2); 
semilogy(SNR_dB_range, rmse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT', 'LineWidth', 2); 
semilogy(SNR_dB_range, crlb_freq_theory_rmse, 'm--', 'DisplayName', 'CRLB', 'LineWidth', 2);

xlabel('SNR (dB)', 'FontSize', 20);
ylabel('RMSE (Hz)', 'FontSize', 20); 
legend('show');
grid on;
box on;
hold off;

%---
% 获取当前图表的坐标轴句柄
ax = gca;
% 调整坐标轴线条颜色和粗细
ax.XColor = 'k'; 
ax.YColor = 'k'; 
ax.LineWidth = 1.5; 
ax.FontSize = 20; 


% 创建新的图窗进行局部放大
figure;
hold on;

% 绘制CZT峰值和二次插值的曲线
semilogy(SNR_dB_range, rmse_czt_peak_only, 'g-s', 'DisplayName', 'CZT', 'LineWidth', 2);
semilogy(SNR_dB_range, rmse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT', 'LineWidth', 2); 
semilogy(SNR_dB_range, crlb_freq_theory_rmse, 'm--', 'DisplayName', 'CRLB', 'LineWidth', 2); 

xlabel('SNR (dB)', 'FontSize', 20);
ylabel('RMSE (Hz)', 'FontSize', 20); 
legend('show');
grid on;
box on;
hold off;

%---
% 获取当前图表的坐标轴句柄
ax = gca;
% 调整坐标轴线条颜色和粗细
ax.XColor = 'k';
ax.YColor = 'k';
ax.LineWidth = 1.2;
ax.FontSize = 20;


%% 5. 导出数据为CSV文件 ✅ 新增部分
% 创建表格
results_table = table(...
    SNR_dB_range', ...
    rmse_fft_peak, ...
    rmse_macleod, ...
    rmse_czt_peak_only, ...
    rmse_czt_quad, ...
    crlb_freq_theory_rmse, ...
    'VariableNames', {...
        'SNR_dB', ...
        'RMSE_FFT_Hz', ...
        'RMSE_Macleod_Hz', ...
        'RMSE_CZT_Peak_Only_Hz', ...
        'RMSE_Macleod_CZT_Hz', ...
        'CRLB_RMSE_Hz' ...
    }...
);

% 指定文件名（可修改路径）
filename = 'SNR_vs_RMSE_results.csv';

% 写入CSV
%writetable(results_table, filename, 'WriteRowNames', false);

% 显示提示
disp(['✅ 仿真完成，数据已保存至: ', pwd, '/', filename]);

%% Macleod算法函数 (保持不变)
function [f_est, delta, peak_mag] = macleod_algorithm(x, Fs, N)
    X = fft(x);
    X_abs_sq = abs(X).^2;
    [~, k0] = max(X_abs_sq); 
    k0 = k0(1);
    
    % 边界保护
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