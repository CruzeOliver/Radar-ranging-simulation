%% 清空环境
clc; clear; close all;

%% 1. 读取CSV文件
filename = "D:\code\python\Andar_UDP_PY\0916岸达测\2920.csv"; 

% 读取数据（包含表头）
data = readtable(filename);

% 提取第3-6列（四种算法的测距值），去掉单位 "m"
distance_data = data{:, 3:6};
if iscell(distance_data)   % 如果是单元格，先转成数字
    distance_data = str2double(erase(distance_data, "m"));
end

%% 2. 定义真实距离
real_distance = 2.920; % 单位：米
real_distance_vector = repmat(real_distance, size(distance_data,1), 1);

%% 3. 计算绝对误差（mm）
error_fft         = abs(distance_data(:,1) - real_distance_vector) * 1000; % mm
error_macleod     = abs(distance_data(:,2) - real_distance_vector) * 1000;
error_czt         = abs(distance_data(:,3) - real_distance_vector) * 1000;
error_macleod_czt = abs(distance_data(:,4) - real_distance_vector) * 1000;

% 合并误差矩阵
error_matrix = [error_fft, error_macleod, error_czt, error_macleod_czt];

%% 4. 绘制误差曲线，统一线条和符号
figure('Name','绝对误差 (mm)','Color','w');

% 每条曲线设置不同的线条样式和标记
markers = {'-o','-^','-s','-d'};  % 红色圆圈、蓝色三角形、绿色方块、黑色菱形
h = gobjects(4,1);
h(1) = plot(error_matrix(:,1), 'r-o', 'LineWidth', 2, 'DisplayName', 'FFT');
hold on;
h(2) = plot(error_matrix(:,2), 'b-^', 'LineWidth', 2, 'DisplayName', 'Macleod');
h(3) = plot(error_matrix(:,3), 'g-s', 'LineWidth', 2, 'DisplayName', 'CZT');
h(4) = plot(error_matrix(:,4), 'k-d', 'LineWidth', 2, 'DisplayName', 'Macleod-CZT');

% 设置标题和坐标轴
%title('绝对误差 (mm)','FontSize',14);
xlabel('Measurement number','FontSize',12);
ylabel('Distance offset (mm)','FontSize',12);

% 设置y轴范围（自动收紧到数据范围附近，留一些边距）
emin = min(error_matrix(:),[],'omitnan'); 
emax = max(error_matrix(:),[],'omitnan');
pad = max(0.05*(emax-emin), 0.5);  % 至少留0.5mm的边距
ylim([emin-pad, emax+pad]);

% 图例
legend(h, {'FFT', 'Macleod', 'CZT', 'Macleod-CZT'}, 'Location','southoutside','Orientation','horizontal');

grid on;

disp('✅ 绝对误差曲线绘制完成 (单位: mm)');
