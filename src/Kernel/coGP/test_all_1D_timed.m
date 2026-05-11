clear; close all; clc;
rng(10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FORRESTER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hf = @(x) (6*x - 2).^2 .* sin(12*x - 4);

A = 0.5; B = 10; C = 5;
lf = @(x) A*hf(x) + B*(x - 0.5) - C;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAINING DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xe = [0; 0.4; 0.6; 1];
ye = hf(Xe);

Xc = linspace(0,1,11)';
yc = lf(Xc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST GRID
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(0,1,200)';   % finer grid for evaluation
true = hf(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 1. GPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gpr = GPR(Xe, ye, [], true);

t_gpr_opt = timeit(@() gpr.optimize());
t_gpr_inf = timeit(@() gpr.inference(x));

[mg, ~] = gpr.inference(x);
rmse_gpr = sqrt(mean((mg - true).^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 2. CoGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ck = coGPR(Xc, Xe, yc, ye);

t_ck_opt = timeit(@() ck.optimize());
t_ck_inf = timeit(@() ck.inference(x));

[mck, ~] = ck.inference(x);
rmse_ck = sqrt(mean((mck - true).^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 3. multiGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mf = multiGPR(Xc, Xe, yc, ye);

t_mf_opt = timeit(@() mf.optimize());
t_mf_inf = timeit(@() mf.inference(x));

[mf_mean, ~] = mf.inference(x);
rmse_mf = sqrt(mean((mf_mean - true).^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== PLOTTING =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; hold on;

plot(x, lf(x), 'r', 'LineWidth', 1.5);
plot(x, hf(x), 'b', 'LineWidth', 2);

scatter(Xc, yc, 60, '^', 'filled');
scatter(Xe, ye, 80, 'ks', 'filled');

plot(x, mg, 'k--', 'LineWidth', 1.5);
plot(x, mck, 'm-', 'LineWidth', 1.5);
plot(x, mf_mean, 'g:', 'LineWidth', 2);

legend('Low fidelity','High fidelity','Low samples','High samples', ...
       'GPR','CoGPR','multiGPR','Location','best');

title('Forrester Multi-Fidelity Benchmark');
xlabel('x'); ylabel('f(x)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== RESULTS =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('RMSE\n');
fprintf('====================================\n');
fprintf('GPR       : %.6e\n', rmse_gpr);
fprintf('CoGPR     : %.6e\n', rmse_ck);
fprintf('multiGPR  : %.6e\n', rmse_mf);

fprintf('\n====================================\n');
fprintf('OPTIMIZATION TIME (s)\n');
fprintf('====================================\n');
fprintf('GPR       : %.4f\n', t_gpr_opt);
fprintf('CoGPR     : %.4f\n', t_ck_opt);
fprintf('multiGPR  : %.4f\n', t_mf_opt);

fprintf('\n====================================\n');
fprintf('INFERENCE TIME (s)\n');
fprintf('====================================\n');
fprintf('GPR       : %.4f\n', t_gpr_inf);
fprintf('CoGPR     : %.4f\n', t_ck_inf);
fprintf('multiGPR  : %.4f\n', t_mf_inf);