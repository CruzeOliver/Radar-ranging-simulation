clear;
close all;
clc;

%% 1. 参数设置
c = 3e8;            % 光速
Fs = 7.14e6;        % 采样频率
T_chirp = 98e-6;    % Chirp周期
Bw = 3000e6;        % 雷达带宽
N = 256;            % FFT点数
f_true = 628000;   % 真实频率408000
n_monte = 1000;     % Monte Carlo仿真次数
M = 32;             % Chirp-Z点数
B_fft_res = Fs / N; % FFT的分辨率


%% 2. 定义仿真范围与结果存储
SNR_dB_range = -5:1:20; % 信噪比（dB）范围
n_snr_points = length(SNR_dB_range);

% 初始化MSE记录矩阵，每一行对应一个SNR点
mse_fft_peak = zeros(n_snr_points, 1);
mse_macleod = zeros(n_snr_points, 1);
mse_czt_peak_only = zeros(n_snr_points, 1);
mse_czt_quad = zeros(n_snr_points, 1);
crlb_freq_theory = zeros(n_snr_points, 1);

%% 3. 蒙特卡洛仿真主循环
% 外层循环：遍历不同的信噪比
for i = 1:n_snr_points
    current_SNR_dB = SNR_dB_range(i);
    snr_linear = 10^(current_SNR_dB / 10);
    
    % 计算当前SNR下的CRLB理论值
    %crlb_freq_theory(i) = (3 * Fs^2) / (2 * pi^2 * N^3 * snr_linear);
    B = 1 * B_fft_res;  % CZT 带宽，例如设为 2 个 FFT bin
    %crlb_freq_theory(i) = 1 / (2 * pi^2 * snr_linear * (M / B)^2);
    crlb_freq_theory(i) =  3 *Fs^2 / (8 * pi^2 * snr_linear * (M^3 + N^3));
    %fprintf('在SNR = %d dB下的CRLB频率下界为：%.6f Hz^2\n', SNR_dB, crlb_freq_theory);
    
    % 初始化当前SNR下的临时误差记录数组
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
        
         %% Step 2: Chirp-Z变换 (CZT)
        % Macleod算法得到的频率作为CZT的搜索中心
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

        %% Step 2: Chirp-Z变换 (CZT)
        % Macleod算法得到的频率作为CZT的搜索中心
        f_start = f_macleod - B_fft_res / 2;
        f_step = B_fft_res / M;
        f_axis = f_start + (0:M-1) * f_step;
        w = exp(-1j * 2 * pi * f_step / Fs);
        a = exp(1j * 2 * pi * f_start / Fs);
        X_czt = czt(s_noisy, M, w, a);
        [~, k_czt_peak] = max(abs(X_czt));
        
        % 确保索引在有效范围内
        k_czt_peak = max(2, min(k_czt_peak, length(X_czt)-1));
        
        % 仅用峰值位置估计频率
        %f_czt_peak_only = f_axis(k_czt_peak);
        
        %% Step 3: CZT二次插值
        mag_km1 = abs(X_czt(k_czt_peak - 1));
        mag_k0  = abs(X_czt(k_czt_peak));
        mag_kp1 = abs(X_czt(k_czt_peak + 1));
        denom = mag_km1 - 2 * mag_k0 + mag_kp1;
        delta_czt_quad = 0;
        if denom ~= 0
            delta_czt_quad = 0.5 * (mag_km1 - mag_kp1) / denom;
        end
        f_czt_quad = f_axis(k_czt_peak) + delta_czt_quad * f_step;
        
        %% 记录当前迭代的频率误差
        temp_mse_fft(monte) = (f_fft_peak - f_true)^2;
        temp_mse_macleod(monte) = (f_macleod - f_true)^2;
        temp_mse_czt_peak_only(monte) = (f_czt_peak_only - f_true)^2;
        temp_mse_czt_quad(monte) = (f_czt_quad - f_true)^2;
    end
    
    % 计算当前SNR下的平均MSE，并存入主矩阵
    mse_fft_peak(i) = mean(temp_mse_fft);
    mse_macleod(i) = mean(temp_mse_macleod);
    mse_czt_peak_only(i) = mean(temp_mse_czt_peak_only);
    mse_czt_quad(i) = mean(temp_mse_czt_quad);
end

%% 4. 绘制结果
figure;
hold on;

% 使用semilogy绘制半对数曲线，更清晰
semilogy(SNR_dB_range, mse_fft_peak, 'r-o', 'DisplayName', 'FFT', 'LineWidth', 2);
semilogy(SNR_dB_range, mse_macleod, 'b-^', 'DisplayName', 'Macleod', 'LineWidth', 2);
semilogy(SNR_dB_range, mse_czt_peak_only, 'g-s', 'DisplayName', 'CZT', 'LineWidth', 2);
semilogy(SNR_dB_range, mse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT', 'LineWidth', 2);
semilogy(SNR_dB_range, crlb_freq_theory, 'm--', 'DisplayName', 'CRLB', 'LineWidth', 2);

xlabel('SNR(dB)', 'FontSize', 20);
ylabel('MSE(Hz^2)', 'FontSize', 20);
%title('不同算法的均方误差 (MSE) 对比', 'FontSize', 20);
legend('show');
grid on;
box on;
hold off;

%---
% 获取当前图表的坐标轴句柄
ax = gca;
% 调整坐标轴线条颜色和粗细
ax.XColor = 'k'; % 将X轴颜色设置为黑色
ax.YColor = 'k'; % 将Y轴颜色设置为黑色
ax.LineWidth = 1.5; % 设置坐标轴线条粗细
ax.FontSize = 20; % 设置刻度字体大小


% 创建新的图窗进行局部放大
figure;
hold on;

% 绘制CZT峰值和二次插值的曲线
semilogy(SNR_dB_range, mse_czt_peak_only, 'g-s', 'DisplayName', 'CZT', 'LineWidth', 2);
semilogy(SNR_dB_range, mse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT', 'LineWidth', 2);
semilogy(SNR_dB_range, crlb_freq_theory, 'm--', 'DisplayName', 'CRLB', 'LineWidth', 2);

xlabel('SNR(dB)', 'FontSize', 20);
ylabel('MSE(Hz^2)', 'FontSize', 20);
%title('CZT峰值与CZT二次插值性能局部放大对比', 'FontSize', 20);
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