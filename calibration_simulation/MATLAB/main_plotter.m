function main_plotter()
% MAIN_PLOTTER  Calibration scheme comparison simulation
% MATLAB R2021b compatible. English labels as requested.
%
% Saves: mse_vs_observations.png, results_vs_observations.csv (and other
% plots if enabled).

%% --- 0. Environment & reproducibility ---
rng(0);
warning('off','MATLAB:singularMatrix');

%% --- 1. Simulation parameters ---
K = 3;          % number of Tx
L = 4;          % number of Rx
M = K * L;

% ILS parameters
ILS_ITERATIONS = 15; 
ILS_TOLERANCE = 1e-6; 

% Monte Carlo
N_MONTE_CARLO = 50; 

%% --- 2. physical parameters ---
f_c = 77e9;
c = 3e8;
lambda_c = c / f_c;
d_rx = lambda_c / 2;
d_tx = L * d_rx;

%% --- 3. fixed simulation settings (same as python script) ---
I_FIXED = 6;      % (I < M-1 = 11)
SNR_FIXED1 = 20;
SNR_FIXED2 = 15;
ERROR_FIXED = 0.2; % std dev

fprintf('====== Starting Calibration Scheme Comparison Simulation ======\n');
fprintf('Fixed parameters: K=%d, L=%d, M=%d, Monte Carlo Runs=%d\n', K, L, M, N_MONTE_CARLO);

% plot_vs_snr(K,L,I_FIXED,ERROR_FIXED); 
plot_vs_observations(K,L,SNR_FIXED1,ERROR_FIXED, N_MONTE_CARLO, d_tx, d_rx, lambda_c);
% plot_vs_error_magnitude(K,L,SNR_FIXED2,I_FIXED);

fprintf('\n====== Simulation Finished ======\n');
fprintf('All plots (.png) and data files (.csv) have been saved to the working directory.\n');

end

%% ---------------- Subfunctions ----------------

function a = get_virtual_steering_vector(phi, k, l, d_tx, d_rx, lambda_c)
% compute a single virtual steering element
phase = -2j * pi * (k * d_tx + l * d_rx) * sin(phi) / lambda_c;
a = exp(phase);
end

function H = get_full_steering_matrix(phi_array, K, L, d_tx, d_rx, lambda_c)
% generate M x I ideal steering matrix
M = K * L;
I = numel(phi_array);
H = zeros(M, I);
col = 1;
for i = 1:I
    phi = phi_array(i);
    m = 1;
    for k = 0:K-1
        for l = 0:L-1
            H(m,i) = get_virtual_steering_vector(phi, k, l, d_tx, d_rx, lambda_c);
            m = m + 1;
        end
    end
end
end

function phi_est = simple_doa_estimator(x_calibrated, M, K, L, d_tx, d_rx, lambda_c)
% simple grid-search DoA estimator (used inside ILS)
angle_grid = linspace(-pi/2, pi/2, 361);
spectrum = zeros(size(angle_grid));
for idx = 1:numel(angle_grid)
    phi = angle_grid(idx);
    a_phi = get_full_steering_matrix(phi, K, L, d_tx, d_rx, lambda_c);
    a_phi = a_phi(:);
    spectrum(idx) = abs(a_phi' * x_calibrated)^2;
end
[~, peak_index] = max(spectrum);
phi_est = angle_grid(peak_index);
end

function [crb_va, crb_mimo] = calculate_crb(K, L, I, SNR_dB)
% compute CRB (VA and MIMO) as in the python code
M = K * L; %#ok<NASGU>
snr_linear = 10^(SNR_dB / 10);
sum_snr_term = I * snr_linear;
crb_va = 1.0 / (I + sum_snr_term);
crb_mimo = ((K + L) / (K * L)) * crb_va;
end

function gamma_va_est = calibrate_va(Kappa_noisy, H_ideal_known)
% VA calibration (angle-known)
[M, I] = size(H_ideal_known);
P_va = Kappa_noisy(2:end, :) ./ Kappa_noisy(1, :);
H_va = H_ideal_known(2:end, :) ./ H_ideal_known(1, :);

gamma_va_est_partial = zeros(M-1, 1);
for m = 1:(M-1)
    p_m_vec = P_va(m, :)';
    h_m_vec = H_va(m, :)';
    % regularized least squares for scalar gamma: gamma = (h^H h + eps)^-1 h^H p
    numerator = h_m_vec' * p_m_vec;
    denominator = h_m_vec' * h_m_vec + 1e-6;
    gamma_m = numerator / denominator;
    gamma_va_est_partial(m) = gamma_m;
end
gamma_va_est = [1.0; gamma_va_est_partial];
end

function [gamma_tx_est, gamma_rx_est] = calibrate_mimo(Kappa_noisy, phi_known, K, L, d_tx, d_rx, lambda_c)
% MIMO independent calibration (angles known)
I = numel(phi_known);
Kappa_reshaped = reshape(Kappa_noisy, [K, L, I]);

gamma_tx_est = ones(K,1);
for k = 2:K
    A_tx = [];
    b_tx = [];
    for i = 1:I
        phi = phi_known(i);
        h_ki = get_virtual_steering_vector(phi, k-1, 0, d_tx, d_rx, lambda_c) / ...
               get_virtual_steering_vector(phi, 0, 0, d_tx, d_rx, lambda_c);
        for l = 1:L
            p_kli = Kappa_reshaped(k, l, i) / Kappa_reshaped(1, l, i);
            A_tx = [A_tx; h_ki]; %#ok<AGROW>
            b_tx = [b_tx; p_kli]; %#ok<AGROW>
        end
    end
    if ~isempty(A_tx)
        % Solve scalar regression with regularization
        gamma_k = A_tx \ b_tx;
        gamma_tx_est(k) = gamma_k(1);
    else
        gamma_tx_est(k) = 1.0;
    end
end

gamma_rx_est = ones(L,1);
for l = 2:L
    A_rx = [];
    b_rx = [];
    for i = 1:I
        phi = phi_known(i);
        h_li = get_virtual_steering_vector(phi, 0, l-1, d_tx, d_rx, lambda_c) / ...
               get_virtual_steering_vector(phi, 0, 0, d_tx, d_rx, lambda_c);
        for k = 1:K
            p_kli = Kappa_reshaped(k, l, i) / Kappa_reshaped(k, 1, i);
            A_rx = [A_rx; h_li]; %#ok<AGROW>
            b_rx = [b_rx; p_kli]; %#ok<AGROW>
        end
    end
    if ~isempty(A_rx)
        gamma_l = A_rx \ b_rx;
        gamma_rx_est(l) = gamma_l(1);
    else
        gamma_rx_est(l) = 1.0;
    end
end
end

function [gamma_tx_ils, gamma_rx_ils] = calibrate_ils(Kappa_noisy, K, L, d_tx, d_rx, lambda_c)
% ILS self-calibration when angles unknown
[M, I] = size(Kappa_noisy);
ILS_ITERATIONS = 15;
ILS_TOLERANCE = 1e-6;

gamma_tx_ils = ones(K,1);
gamma_rx_ils = ones(L,1);
phi_ils = zeros(I,1);

% initial DoA estimates
for i = 1:I
    phi_ils(i) = simple_doa_estimator(Kappa_noisy(:, i), M, K, L, d_tx, d_rx, lambda_c);
end

for iter_n = 1:ILS_ITERATIONS
    [gamma_tx_new, gamma_rx_new] = calibrate_mimo(Kappa_noisy, phi_ils, K, L, d_tx, d_rx, lambda_c);
    gamma_va_ils = kron(gamma_tx_new, gamma_rx_new);
    Kappa_calibrated = Kappa_noisy ./ reshape(gamma_va_ils, [], 1);

    phi_new = zeros(I,1);
    for i = 1:I
        phi_new(i) = simple_doa_estimator(Kappa_calibrated(:, i), M, K, L, d_tx, d_rx, lambda_c);
    end

    error_tx = norm(gamma_tx_new - gamma_tx_ils) / (norm(gamma_tx_ils)+eps);
    error_rx = norm(gamma_rx_new - gamma_rx_ils) / (norm(gamma_rx_ils)+eps);

    gamma_tx_ils = gamma_tx_new;
    gamma_rx_ils = gamma_rx_new;
    phi_ils = phi_new;

    if (error_tx < ILS_TOLERANCE && error_rx < ILS_TOLERANCE)
        break;
    end
end
end

function [mse_va, mse_mimo, mse_ils] = run_single_simulation(K, L, I, SNR_dB, error_std_dev, d_tx, d_rx, lambda_c)
% Perform one Monte-Carlo simulation: data generation -> calibrations -> MSE
M = K * L;

% ground truth gains
gamma_tx_true = ones(K,1);
gamma_rx_true = ones(L,1);

err_tx_real = error_std_dev * randn(K-1,1);
err_tx_imag = error_std_dev * randn(K-1,1);
gamma_tx_true(2:end) = 1.0 + err_tx_real + 1j * err_tx_imag;

err_rx_real = error_std_dev * randn(L-1,1);
err_rx_imag = error_std_dev * randn(L-1,1);
gamma_rx_true(2:end) = 1.0 + err_rx_real + 1j * err_rx_imag;

gamma_va_true = kron(gamma_tx_true, gamma_rx_true);

% fixed equally spaced angles
phi_true = linspace(-pi/4, pi/4, I);
alpha_true = (randn(1,I) + 1j * randn(1,I)) / sqrt(2);
H_ideal_true = get_full_steering_matrix(phi_true, K, L, d_tx, d_rx, lambda_c);

Kappa_clean = zeros(M, I);
for i = 1:I
    Kappa_clean(:, i) = alpha_true(i) * gamma_va_true .* H_ideal_true(:, i);
end

signal_power = mean(abs(Kappa_clean(:)).^2);
noise_power = signal_power / (10^(SNR_dB / 10));
noise = (randn(M, I) + 1j * randn(M, I)) * sqrt(noise_power / 2);
Kappa_noisy = Kappa_clean + noise;

% run three calibration schemes
gamma_va_est = calibrate_va(Kappa_noisy, H_ideal_true);
[gamma_tx_mimo, gamma_rx_mimo] = calibrate_mimo(Kappa_noisy, phi_true, K, L, d_tx, d_rx, lambda_c);
gamma_va_mimo_est = kron(gamma_tx_mimo, gamma_rx_mimo);
[gamma_tx_ils, gamma_rx_ils] = calibrate_ils(Kappa_noisy, K, L, d_tx, d_rx, lambda_c);
gamma_va_ils_est = kron(gamma_tx_ils, gamma_rx_ils);

% evaluate MSE (normalized by first element)
mse_va = mean(abs(gamma_va_est / gamma_va_est(1) - gamma_va_true / gamma_va_true(1)).^2);
mse_mimo = mean(abs(gamma_va_mimo_est / gamma_va_mimo_est(1) - gamma_va_true / gamma_va_true(1)).^2);
mse_ils = mean(abs(gamma_va_ils_est / gamma_va_ils_est(1) - gamma_va_true / gamma_va_true(1)).^2);
end

function plot_vs_observations(K, L, snr_fixed, error_fixed, N_MONTE_CARLO, d_tx, d_rx, lambda_c)
% Plot: MSE vs number of observations I
fprintf('\n--- Generating Plot 2 (vs. Observations I) ---\n');
fprintf('Fixed parameters: SNR = %d dB, Error StdDev = %.3f\n', snr_fixed, error_fixed);
M = K * L;
i_range = (max(K,L)):5:35;
results = zeros(numel(i_range), 5);
start_time = tic;
for idx = 1:numel(i_range)
    obs_I = i_range(idx);
    mse_mc = zeros(N_MONTE_CARLO, 3);
    for mc = 1:N_MONTE_CARLO
        [mse_mc(mc,1), mse_mc(mc,2), mse_mc(mc,3)] = run_single_simulation(K, L, obs_I, snr_fixed, error_fixed, d_tx, d_rx, lambda_c);
    end
    results(idx,1:3) = mean(mse_mc, 1);
    [crb_va, crb_mimo] = calculate_crb(K, L, obs_I, snr_fixed);
    results(idx,4) = crb_va;
    results(idx,5) = crb_mimo;
    fprintf('  I = %d complete (%d/%d)\n', obs_I, idx, numel(i_range));
end

figure;
plot(i_range, results(:,1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)'); hold on;
plot(i_range, results(:,2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
plot(i_range, results(:,3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
plot(i_range, results(:,4), 'k--', 'DisplayName', 'CRB (VA)');
plot(i_range, results(:,5), 'k-', 'DisplayName', 'CRB (MIMO)');

xline(M-1, '--r', sprintf('VA Underdetermined (I < %d)', M-1), 'LabelHorizontalAlignment','left');

xlabel('Number of Independent Observations (I)');
ylabel('Channel Error MSE (log scale)');
title(sprintf('Calibration Performance vs. Observations (SNR=%ddB)', snr_fixed));
set(gca, 'YScale', 'log');
legend('Location','best');
grid on;
ylim([1e-5, max(1e-3, max(results(:,1:3), [], 'all'))]);

saveas(gcf, 'mse_vs_observations.png');
fprintf('Plot 2 saved to: mse_vs_observations.png\n');

% save CSV
csv_filename = 'results_vs_observations.csv';
T = array2table([i_range(:), results], 'VariableNames', {'I_Observations', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'});
writetable(T, csv_filename);
fprintf('Data for Plot 2 saved to: %s (Time: %.2fs)\n', csv_filename, toc(start_time));
end

function plot_vs_snr(K, L, I_fixed, error_fixed)
% (Not enabled by default) MSE vs SNR
fprintf('\n--- Generating Plot 1 (vs. SNR) ---\n');
fprintf('Fixed parameters: I = %d, Error StdDev = %.3f\n', I_fixed, error_fixed);

snr_range = linspace(0, 25, 10);
N_MONTE_CARLO = 50;
results = zeros(numel(snr_range), 5);
start_time = tic;
for i = 1:numel(snr_range)
    snr = snr_range(i);
    mse_mc = zeros(N_MONTE_CARLO, 3);
    for mc = 1:N_MONTE_CARLO
        [mse_mc(mc,1), mse_mc(mc,2), mse_mc(mc,3)] = run_single_simulation(K, L, I_fixed, snr, error_fixed, L * (lambda_c/2), lambda_c/2, lambda_c);
    end
    results(i,1:3) = mean(mse_mc, 1);
    [crb_va, crb_mimo] = calculate_crb(K, L, I_fixed, snr);
    results(i,4) = crb_va;
    results(i,5) = crb_mimo;
    fprintf('  SNR = %.1f dB complete (%d/%d)\n', snr, i, numel(snr_range));
end

figure;
plot(snr_range, results(:,1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)'); hold on;
plot(snr_range, results(:,2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
plot(snr_range, results(:,3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
plot(snr_range, results(:,4), 'k--', 'DisplayName', 'CRB (VA)');
plot(snr_range, results(:,5), 'k-', 'DisplayName', 'CRB (MIMO)');

xlabel('Signal-to-Noise Ratio (SNR) / dB');
ylabel('Channel Error MSE (log scale)');
title(sprintf('Calibration Performance vs. SNR (I=%d, K=%d, L=%d)', I_fixed, K, L));
set(gca, 'YScale', 'log');
legend('Location','best');
grid on;
ylim([1e-5, max(1e-3, max(results(:,1:3), [], 'all'))]);

saveas(gcf, 'mse_vs_snr.png');
fprintf('Plot 1 saved to: mse_vs_snr.png\n');

% save CSV
csv_filename = 'results_vs_snr.csv';
T = array2table([snr_range(:), results], 'VariableNames', {'SNR_dB', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'});
writetable(T, csv_filename);
fprintf('Data for Plot 1 saved to: %s (Time: %.2fs)\n', csv_filename, toc(start_time));
end

function plot_vs_error_magnitude(K, L, snr_fixed, i_fixed)
% MSE vs error variance
fprintf('\n--- Generating Plot 3 (vs. True Error Variance) ---\n');
fprintf('Fixed parameters: SNR = %d dB, I = %d\n', snr_fixed, i_fixed);

variance_range = linspace(0.05, 0.5, 10);
results = zeros(numel(variance_range), 5);
[crb_va, crb_mimo] = calculate_crb(K, L, i_fixed, snr_fixed);
results(:,4) = crb_va; results(:,5) = crb_mimo;

N_MONTE_CARLO = 50;
start_time = tic;
for idx = 1:numel(variance_range)
    variance = variance_range(idx);
    std_dev = sqrt(variance);
    mse_mc = zeros(N_MONTE_CARLO, 3);
    for mc = 1:N_MONTE_CARLO
        [mse_mc(mc,1), mse_mc(mc,2), mse_mc(mc,3)] = run_single_simulation(K, L, i_fixed, snr_fixed, std_dev, L * (lambda_c/2), lambda_c/2, lambda_c);
    end
    results(idx,1:3) = mean(mse_mc, 1);
    fprintf('   Error Variance = %.2f (StdDev = %.2f) complete (%d/%d)\n', variance, std_dev, idx, numel(variance_range));
end

figure;
plot(variance_range, results(:,1), 'o--', 'DisplayName', 'Scheme 1 (VA - Angle Known)'); hold on;
plot(variance_range, results(:,2), 's-', 'DisplayName', 'Scheme 2 (MIMO - Angle Known)');
plot(variance_range, results(:,3), 'x:', 'DisplayName', 'Scheme 3 (ILS - Angle Unknown)');
plot(variance_range, results(:,4), 'k--', 'DisplayName', 'CRB (VA)');
plot(variance_range, results(:,5), 'k-', 'DisplayName', 'CRB (MIMO)');

xlabel('True Channel Error Variance (Error Var, \sigma_{\gamma}^2)');
ylabel('Channel Error MSE (log scale)');
title(sprintf('Calibration Performance vs. True Error Magnitude (SNR=%ddB, I=%d)', snr_fixed, i_fixed));
set(gca, 'YScale', 'log');
legend('Location','best');
grid on;
ylim([1e-5, max(1e-3, max(results(:,1:3), [], 'all'))]);

saveas(gcf, 'mse_vs_error_magnitude.png');
fprintf('Plot 3 saved to: mse_vs_error_magnitude.png\n');

% save CSV
csv_filename = 'results_vs_error_magnitude.csv';
T = array2table([variance_range(:), results], 'VariableNames', {'Error_Variance', 'MSE_VA', 'MSE_MIMO', 'MSE_ILS', 'CRB_VA', 'CRB_MIMO'});
writetable(T, csv_filename);
fprintf('Data for Plot 3 saved to: %s (Time: %.2fs)\n', csv_filename, toc(start_time));
end
