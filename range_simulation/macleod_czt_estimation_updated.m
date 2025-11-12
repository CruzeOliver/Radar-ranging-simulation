%% 1DFFT、Macleod算法、CZT峰值、Macleod+CZT估计对频率和距离的估计
clear; close all; clc;

%% 参数设置
c = 3e8;
Fs = 7.14e6;
T_chirp = 98e-6;
Bw = 3000e6;
N = 256;
f_true = 3777000;
SNR_dB = 5;
M = 32; % Chirp-Z点数
B = 2*Fs/N;
n_monte = 100;

A = 1;  % 幅度为1
snr_linear = 10^(SNR_dB/10);
sigma2 = 1;
%CRLB_freq = 6 * sigma2 * Fs^2 / ( (2*pi)^2 * A^2 * N * (N^2 - 1) * snr_linear );
CRLB_freq = (3 * Fs^2) / (2 * pi^2 * N^3 * snr_linear); 
sqrt(CRLB_freq)

% 初始化误差记录
error_fft_peak = zeros(n_monte,1);
error_macleod = zeros(n_monte,1);
error_czt_peak_only = zeros(n_monte,1);
error_czt_quad = zeros(n_monte,1);

range_error_fft_peak = zeros(n_monte,1);
range_error_macleod = zeros(n_monte,1);
range_error_czt_peak_only = zeros(n_monte,1);
range_error_czt_quad = zeros(n_monte,1);

%% Monte Carlo蒙特卡洛仿真
for monte = 1:n_monte
    t = (0:N-1)'/Fs;
    s = exp(1j*2*pi*f_true*t);
    noise = (randn(size(t)) + 1j*randn(size(t)))/sqrt(2);
    s_noisy = s*sqrt(snr_linear) + noise;
    
    %% Step 0: FFT频率估计
    X_fft = fft(s_noisy);
    [~, k_fft_peak] = max(abs(X_fft));
    f_fft_peak = (k_fft_peak - 1) * Fs / N;

    %% Step 1: Macleod粗估
    [f_macleod_coarse, ~, ~] = macleod_algorithm(s_noisy, Fs, N);

    %% Step 2: Chirp-Z
    f_start = f_macleod_coarse - B/2;
    f_step = B / M;
    f_axis = f_start + (0:M-1) * f_step;
    w = exp(-1j*2*pi*f_step/Fs);
    a = exp(1j*2*pi*f_start/Fs);
    X_czt = czt(s_noisy, M, w, a);
    [~, k_czt_peak] = max(abs(X_czt));
    k_czt_peak = max(2, min(k_czt_peak, length(X_czt)-1)); % 边界保护
    f_czt_peak_only = f_axis(k_czt_peak);

    %% Step 3: CZT二次插值
    mag_km1 = abs(X_czt(k_czt_peak-1));
    mag_k0  = abs(X_czt(k_czt_peak));
    mag_kp1 = abs(X_czt(k_czt_peak+1));
    denom = mag_km1 - 2*mag_k0 + mag_kp1;
    delta_czt_quad = 0;
    if denom ~= 0
        delta_czt_quad = 0.5 * (mag_km1 - mag_kp1) / denom;
    end
    f_czt_quad = f_axis(k_czt_peak) + delta_czt_quad * f_step;

    %% 距离估计
    R_true = c * f_true * T_chirp / (2 * Bw);
    R_fft_peak = c * f_fft_peak * T_chirp / (2 * Bw);
    R_macleod = c * f_macleod_coarse * T_chirp / (2 * Bw);
    R_czt_peak_only = c * f_czt_peak_only * T_chirp / (2 * Bw);
    R_czt_quad = c * f_czt_quad * T_chirp / (2 * Bw);

    %% 误差记录
    error_fft_peak(monte) = f_fft_peak - f_true;
    error_macleod(monte) = f_macleod_coarse - f_true;
    error_czt_peak_only(monte) = f_czt_peak_only - f_true;
    error_czt_quad(monte) = f_czt_quad - f_true;

    range_error_fft_peak(monte) = R_fft_peak - R_true;
    range_error_macleod(monte) = R_macleod - R_true;
    range_error_czt_peak_only(monte) = R_czt_peak_only - R_true;
    range_error_czt_quad(monte) = R_czt_quad - R_true;

    fprintf('Iter %d: FFT=%.2f, Macleod=%.2f, CZT=%.2f, CZT-Quad=%.2f\n', ...
        monte, f_fft_peak, f_macleod_coarse, f_czt_peak_only, f_czt_quad);
        % 仅前2次Monte Carlo画图验证频谱
    if monte <= 2
        figure;
        f_axis_fft = (0:N-1) * Fs / N;
        plot(f_axis_fft, abs(fft(s_noisy)), 'LineWidth', 1.2); hold on;
        xline(f_true, 'r--', 'True f', 'LabelVerticalAlignment','bottom');
        xline(f_macleod_coarse, 'b--', 'Macleod', 'LabelVerticalAlignment','middle');
        xline(f_czt_quad, 'g--', 'CZT-Quad', 'LabelVerticalAlignment','top');
        legend('FFT谱', '真实频率', 'Macleod估计', 'CZT插值估计');
        xlabel('频率 (Hz)');
        ylabel('幅度');
        title(sprintf('Monte Carlo #%d 频谱图对比', monte));
        grid on;
    end

end

%% RMSE计算
rmse_fft_peak = sqrt(mean(error_fft_peak.^2));
rmse_macleod = sqrt(mean(error_macleod.^2));
rmse_czt_peak_only = sqrt(mean(error_czt_peak_only.^2));
rmse_czt_quad = sqrt(mean(error_czt_quad.^2));

rmse_range_fft_peak = sqrt(mean(range_error_fft_peak.^2));
rmse_range_macleod = sqrt(mean(range_error_macleod.^2));
rmse_range_czt_peak_only = sqrt(mean(range_error_czt_peak_only.^2));
rmse_range_czt_quad = sqrt(mean(range_error_czt_quad.^2));

%% 可视化结果
figure('Position',[100,100,1600,800]);
subplot(2,2,1);
plot(1:n_monte, abs(error_fft_peak), 'k--', ...
     1:n_monte, abs(error_macleod), 'b-', ...
     1:n_monte, abs(error_czt_peak_only), 'r-', ...
     1:n_monte, abs(error_czt_quad), 'g-','LineWidth',1.5); hold on;
yline(sqrt(CRLB_freq), 'k--', 'CRLB', 'LineWidth', 1.5);
xlabel('Monte Carlo 次数'); ylabel('频率误差幅值(Hz)');
legend('FFT','Macleod','CZT峰值','CZT二次');
title(['RMSE频率估计误差对比 (SNR=',num2str(SNR_dB),' dB)']);
grid on;

subplot(2,2,2);
plot(1:n_monte, abs(range_error_fft_peak), 'k--', ...
     1:n_monte, abs(range_error_macleod), 'b-', ...
     1:n_monte, abs(range_error_czt_peak_only), 'r-', ...
     1:n_monte, abs(range_error_czt_quad), 'g-','LineWidth',1.5);
xlabel('Monte Carlo 次数'); ylabel('距离误差(m)');
legend('FFT','Macleod','CZT峰值','CZT二次');
title('距离估计误差对比');
grid on;

subplot(2,2,3);
histogram(abs(error_fft_peak),20,'Normalization','probability'); hold on;
histogram(abs(error_macleod),20,'Normalization','probability');
histogram(abs(error_czt_peak_only),20,'Normalization','probability');
histogram(abs(error_czt_quad),20,'Normalization','probability');
xlabel('频率误差幅值(Hz)'); ylabel('概率密度');
legend('FFT','Macleod','CZT峰值','CZT二次');
title('频率误差分布');
hold off;

subplot(2,2,4);
bar([rmse_fft_peak, rmse_macleod, rmse_czt_peak_only, rmse_czt_quad]);
set(gca,'XTickLabel',{'FFT','Macleod','CZT峰值','CZT二次'});
ylabel('RMSE频率误差(Hz)');
title('算法平均频率误差比较');
grid on;

%% 输出性能统计
fprintf('\n======= 算法性能统计 =======\n');
fprintf('FFT                - RMSE频率误差: %.6f Hz, RMSE距离误差: %.6f m\n', ...
        rmse_fft_peak, rmse_range_fft_peak);
fprintf('Macleod (粗估)     - RMSE频率误差: %.6f Hz, RMSE距离误差: %.6f m\n', ...
        rmse_macleod, rmse_range_macleod);
fprintf('CZT (仅峰值)       - RMSE频率误差: %.6f Hz, RMSE距离误差: %.6f m\n', ...
        rmse_czt_peak_only, rmse_range_czt_peak_only);
fprintf('CZT (二次插值)     - RMSE频率误差: %.6f Hz, RMSE距离误差: %.6f m\n', ...
        rmse_czt_quad, rmse_range_czt_quad);

%% Macleod算法函数
function [f_est, delta, peak_mag] = macleod_algorithm(x, Fs, N)
    X = fft(x);
    X_abs_sq = abs(X).^2;
    [~, k0] = max(X_abs_sq); k0 = k0(1);
    if k0 <= 1 || k0 >= N
        warning('峰值靠近FFT边缘，可能影响插值精度');
    end
    k0 = max(2, min(k0, N-1));
    X_km1 = X_abs_sq(k0-1); X_k0 = X_abs_sq(k0); X_kp1 = X_abs_sq(k0+1);
    denom = X_km1 - 2*X_k0 + X_kp1;
    delta = 0;
    if denom ~= 0
        delta = (X_km1 - X_kp1)/(2*denom);
    end
    f_est = (k0 - 1 + delta)*Fs/N;
    peak_mag = abs(X(k0));
end
