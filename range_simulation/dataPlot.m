% 清除工作区
clear; clc; close all;

% 读取CSV文件，保留原始列名
F_true = 239148;
filename = "D:\code\python\Andar_UDP_PY\0923计量测\1124.csv";
data = readtable(filename, 'VariableNamingRule', 'preserve');

% 提取频率数据（注意：含连字符的列名要用 data.('ColName') 方式访问）
fft_freq     = data.('FFT-fre');
macleod_freq = data.('Macleod-fre');
ctz_freq     = data.('CTZ-fre');
mctz_freq    = data.('MCTZ-fre');

% 提取索引（横轴）
index = data.index;

% 创建图形
figure;
hold on;  % 保持所有曲线在同一图中

% 绘制四条曲线，使用您指定的样式
plot(index, fft_freq,     'r-o', 'DisplayName', 'FFT',     'LineWidth', 2);
plot(index, macleod_freq, 'b-^', 'DisplayName', 'Macleod', 'LineWidth', 2);
plot(index, ctz_freq,     'g-s', 'DisplayName', 'CZT',     'LineWidth', 2);
plot(index, mctz_freq,    'k-d', 'DisplayName', 'MCZT',    'LineWidth', 2);

plot([min(index), max(index)], [F_true, F_true], 'm--', 'LineWidth', 2, 'DisplayName', 'True Frequency');

% 添加标题和坐标轴标签
title('Frequency Estimation Comparison Over Time', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Index', 'FontSize', 12);
ylabel('Frequency (Hz)', 'FontSize', 12);

% 添加图例（根据 DisplayName 显示）
legend('show', 'Location', 'best', 'FontSize', 11);

% 添加网格（增强可读性）
grid on;
grid minor;  % 显示细网格

% 可选：设置坐标轴外观
set(gca, 'FontSize', 10);
% xlim([min(index) max(index)]);  % 横轴范围
% ylim([416000 418500]);          % 可根据数据范围手动设置纵轴（可选）

% 美化布局
set(gcf, 'Position', [100, 100, 900, 500]);  % 设置窗口大小
hold off;