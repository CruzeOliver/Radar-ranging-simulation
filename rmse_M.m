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
n_monte = 10000; % Monte Carlo仿真次数
B_fft_res = Fs / N; % FFT的分辨率

% 转换线性信噪比
snr_linear = 10^(SNR_dB / 10);


%% 2. 定义仿真范围与结果存储
% CZT点数变化范围
M_range = 16:16:256; % 从16到256，步长为8
n_m_points = length(M_range);

% 初始化RMSE记录矩阵
rmse_czt_peak_only = zeros(n_m_points, 1); 
rmse_czt_quad = zeros(n_m_points, 1); 
crlb_freq_theory_rmse = zeros(n_m_points, 1); 

% 新增：初始化时间记录矩阵 (单位：秒)
time_czt_peak_only = zeros(n_m_points, 1);
time_czt_quad = zeros(n_m_points, 1);


%% 3. 蒙特卡洛仿真主循环
disp('开始蒙特卡洛仿真...');
% 外层循环：遍历不同的CZT点数
for i = 1:n_m_points
    current_M = M_range(i);
    disp(['  -> 正在仿真 M = ', num2str(current_M), ' ...']);
    
    % 计算当前M值下的CRLB（MSE形式）
    % 注意：原代码中的CRLB公式针对FFT-CZT，若需更通用公式可参考注释替换
    crlb_freq_theory_mse = 3 * Fs^2 / (8 * pi^2 * snr_linear * (current_M^3 + N^3)); 
    % 通用CZT频率估计CRLB参考公式：crlb = (6 * Fs^2) / (8 * pi^2 * snr_linear * N * (N^2 - 1))
    
    % 将CRLB（MSE形式）开方，得到RMSE的CRLB
    crlb_freq_theory_rmse(i) = sqrt(crlb_freq_theory_mse); 
    
    % 初始化当前M值下的临时误差和时间记录数组
    temp_mse_czt_peak_only = zeros(n_monte, 1);
    temp_mse_czt_quad = zeros(n_monte, 1);
    temp_time_czt_peak_only = zeros(n_monte, 1); % 计时存储
    temp_time_czt_quad = zeros(n_monte, 1);     % 计时存储
    
    % 内层循环：在当前M值下重复n_monte次仿真
    for monte = 1:n_monte
        % 生成含噪声信号
        t = (0:N-1)' / Fs;
        s = exp(1j * 2 * pi * f_true * t);
        noise = (randn(size(t)) + 1j * randn(size(t))) / sqrt(2);
        s_noisy = s * sqrt(snr_linear) + noise;

        %% Macleod 定位 (作为 CZT 的前置步骤)
        X_fft = fft(s_noisy);
        [~, k_fft_peak] = max(abs(X_fft));
        f_fft_peak = (k_fft_peak(1) - 1) * Fs / N; % 确保只取一个峰值
        [f_macleod, ~, ~] = macleod_algorithm(s_noisy, Fs, N);
        
        %% Step 2-1: FFT-CZT (纯CZT峰值定位) - 性能和时间测量
        tic; % 计时开始
        f_start = f_fft_peak - B_fft_res / 2;
        f_step = B_fft_res / current_M; 
        f_axis = f_start + (0:current_M-1) * f_step; 
        w = exp(-1j * 2 * pi * f_step / Fs);
        a = exp(1j * 2 * pi * f_start / Fs);
        X_czt = czt(s_noisy, current_M, w, a);
        [~, k_czt_peak_fft] = max(abs(X_czt)); 
        k_czt_peak_fft = max(1, min(k_czt_peak_fft(1), length(X_czt))); % 防止索引越界
        f_czt_peak_only = f_axis(k_czt_peak_fft);
        temp_time_czt_peak_only(monte) = toc; % 计时结束
        
        %% Step 2-2/3: Macleod-CZT (二次插值) - 性能和时间测量
        tic; % 计时开始
        f_start_macleod = f_macleod - B_fft_res / 2;
        f_step_macleod = B_fft_res / current_M; 
        f_axis_macleod = f_start_macleod + (0:current_M-1) * f_step_macleod; 
        w_macleod = exp(-1j * 2 * pi * f_step_macleod / Fs);
        a_macleod = exp(1j * 2 * pi * f_start_macleod / Fs);
        X_czt_macleod = czt(s_noisy, current_M, w_macleod, a_macleod);
        [~, k_czt_peak_macleod] = max(abs(X_czt_macleod)); 
        k_czt_peak_macleod = k_czt_peak_macleod(1); % 确保只取一个峰值
        k_czt_peak_macleod = max(2, min(k_czt_peak_macleod, length(X_czt_macleod)-1)); % 保证邻点存在
        
        % CZT二次插值 (基于 Macleod 定位)
        mag_km1 = abs(X_czt_macleod(k_czt_peak_macleod - 1));
        mag_k0  = abs(X_czt_macleod(k_czt_peak_macleod));
        mag_kp1 = abs(X_czt_macleod(k_czt_peak_macleod + 1));
        denom = mag_km1 - 2 * mag_k0 + mag_kp1;
        delta_czt_quad = 0;
        if denom ~= 0
            delta_czt_quad = 0.5 * (mag_km1 - mag_kp1) / denom;
        end
        f_czt_quad = f_axis_macleod(k_czt_peak_macleod) + delta_czt_quad * f_step_macleod;
        temp_time_czt_quad(monte) = toc; % 计时结束
        
        %% 记录当前迭代的频率平方误差 (MSE)
        temp_mse_czt_peak_only(monte) = (f_czt_peak_only - f_true)^2;
        temp_mse_czt_quad(monte) = (f_czt_quad - f_true)^2;
    end
    
    % 计算当前M值下的平均RMSE，存入主矩阵
    rmse_czt_peak_only(i) = sqrt(mean(temp_mse_czt_peak_only)); 
    rmse_czt_quad(i) = sqrt(mean(temp_mse_czt_quad)); 

    % 计算当前M值下的平均运行时间，存入主矩阵
    time_czt_peak_only(i) = mean(temp_time_czt_peak_only);
    time_czt_quad(i) = mean(temp_time_czt_quad);
end
disp('蒙特卡洛仿真完成。');


%% 4. 绘制结果
% 绘制 RMSE 曲线
figure;
hold on;
semilogy(M_range, rmse_czt_peak_only, 'g-s', 'DisplayName', 'FFT-CZT (RMSE)', 'LineWidth', 2); 
semilogy(M_range, rmse_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT (RMSE)', 'LineWidth', 2); 
semilogy(M_range, crlb_freq_theory_rmse, 'm--', 'DisplayName', 'CRLB (RMSE)', 'LineWidth', 2);

xlabel('Number of CZT Points (M)', 'FontSize', 20);
ylabel('RMSE (Hz)', 'FontSize', 20); 
title('RMSE vs. CZT Points M', 'FontSize', 24);
legend('show');
grid on;
box on;
hold off;

% 绘制运行时间曲线
figure;
hold on;
plot(M_range, time_czt_peak_only, 'g-s', 'DisplayName', 'FFT-CZT (Time)', 'LineWidth', 2);
plot(M_range, time_czt_quad, 'k-d', 'DisplayName', 'Macleod-CZT (Time)', 'LineWidth', 2);

xlabel('Number of CZT Points (M)', 'FontSize', 20);
ylabel('Average Execution Time per Monte Carlo Run (s)', 'FontSize', 20); 
title('Average Execution Time vs. CZT Points M', 'FontSize', 24);
legend('show');
grid on;
box on;
hold off;

% 统一调整坐标轴样式 (适用于所有图窗)
hFigs = findobj('Type','figure');
for fig = hFigs'
    figure(fig);
    ax = gca;
    ax.XColor = 'k'; 
    ax.YColor = 'k'; 
    ax.LineWidth = 1.5; 
    ax.FontSize = 20; 
end


%% 5. 导出数据为 CSV 文件
% 创建结果表格
results_table = table(...
    M_range', ...                      % CZT点数
    rmse_czt_peak_only, ...           % FFT-CZT RMSE
    time_czt_peak_only, ...           % FFT-CZT 平均时间
    rmse_czt_quad, ...                % Macleod-CZT RMSE
    time_czt_quad, ...                % Macleod-CZT 平均时间
    crlb_freq_theory_rmse, ...        % 理论CRLB RMSE
    'VariableNames', {...
        'M_CZT_Points', ...
        'RMSE_FFT_CZT_Hz', ...
        'Time_FFT_CZT_s', ...
        'RMSE_Macleod_CZT_Hz', ...
        'Time_Macleod_CZT_s', ...
        'CRLB_RMSE_Hz' ...
    }...
);

% 指定输出文件名
filename = 'M_vs_Performance_results.csv';

% 写入CSV文件
writetable(results_table, filename, 'WriteRowNames', false);

% 显示提示信息
disp(['✅ 仿真完成，结果已保存至: ', pwd, '/', filename]);

%% Macleod算法函数（需与主脚本在同一目录）
function [f_est, delta, peak_mag] = macleod_algorithm(x, Fs, N)
    X = fft(x);
    X_abs_sq = abs(X).^2;
    [~, k0] = max(X_abs_sq);
    k0 = k0(1); % 确保只取一个索引
    
    k0 = max(2, min(k0, N-1)); % 防止邻点越界
    
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