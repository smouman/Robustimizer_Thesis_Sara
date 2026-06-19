clear; close all; clc;
rng(10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TOTAL TIMER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_total_start = tic;

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

fprintf('\nBuilding GPR model...\n');

t_gpr_build_start = tic;

gpr = GPR(Xe, ye, [], true);

t_gpr_build = toc(t_gpr_build_start);

fprintf('Done.\n');

fprintf('Optimizing GPR model...\n');

t_gpr_opt = timeit(@() gpr.optimize());

fprintf('Running GPR inference...\n');

t_gpr_inf = timeit(@() gpr.inference(x));

[mg, vg] = gpr.inference(x);

rmse_gpr = sqrt(mean((mg - true).^2));

t_gpr_total = t_gpr_build + t_gpr_opt + t_gpr_inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 2. CoGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBuilding CoGPR model...\n');

t_ck_build_start = tic;

ck = coGPR(Xc, Xe, yc, ye);

t_ck_build = toc(t_ck_build_start);

fprintf('Done.\n');

fprintf('Optimizing CoGPR model...\n');

t_ck_opt = timeit(@() ck.optimize());

fprintf('Running CoGPR inference...\n');

t_ck_inf = timeit(@() ck.inference(x));

[mck, vck] = ck.inference(x);

rmse_ck = sqrt(mean((mck - true).^2));

t_ck_total = t_ck_build + t_ck_opt + t_ck_inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 3. multiGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBuilding multiGPR model...\n');

t_mf_build_start = tic;

mf = multiGPR(Xc, Xe, yc, ye);

t_mf_build = toc(t_mf_build_start);

fprintf('Done.\n');

fprintf('Optimizing multiGPR model...\n');

t_mf_opt = timeit(@() mf.optimize());

fprintf('Running multiGPR inference...\n');

t_mf_inf = timeit(@() mf.inference(x));

[mf_mean, vmf] = mf.inference(x);


rmse_mf = sqrt(mean((mf_mean - true).^2));

t_mf_total = t_mf_build + t_mf_opt + t_mf_inf;


std_gpr = sqrt(max(diag(vg),0));
std_ck  = sqrt(max(diag(vck),0));
std_mf  = sqrt(max(diag(vmf),0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLOBAL TOTAL TIME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_total = toc(t_total_start);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD TIMES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('MODEL BUILD TIME (s)\n');
fprintf('====================================\n');

fprintf('GPR       : %.6f\n', t_gpr_build);
fprintf('CoGPR     : %.6f\n', t_ck_build);
fprintf('multiGPR  : %.6f\n', t_mf_build);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIMIZATION TIMES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('OPTIMIZATION TIME (s)\n');
fprintf('====================================\n');

fprintf('GPR       : %.6f\n', t_gpr_opt);
fprintf('CoGPR     : %.6f\n', t_ck_opt);
fprintf('multiGPR  : %.6f\n', t_mf_opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INFERENCE TIMES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('INFERENCE TIME (s)\n');
fprintf('====================================\n');

fprintf('GPR       : %.6f\n', t_gpr_inf);
fprintf('CoGPR     : %.6f\n', t_ck_inf);
fprintf('multiGPR  : %.6f\n', t_mf_inf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TOTAL MODEL TIMES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('TOTAL MODEL TIME (s)\n');
fprintf('====================================\n');

fprintf('GPR       : %.6f\n', t_gpr_total);
fprintf('CoGPR     : %.6f\n', t_ck_total);
fprintf('multiGPR  : %.6f\n', t_mf_total);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OVERALL SCRIPT TIME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('OVERALL SCRIPT TIME (s)\n');
fprintf('====================================\n');

fprintf('Total Runtime : %.6f\n', t_total);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== PLOTTING =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,1,1); hold on;

fill([x; flipud(x)], ...
     [mg + 1.96*std_gpr; flipud(mg - 1.96*std_gpr)], ...
     [0.85 0.85 0.85], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.5);

plot(x, true, 'b', 'LineWidth', 2);
plot(x, mg, 'k--', 'LineWidth', 2);

scatter(Xe, ye, 70, 'ks', 'filled');

title('GPR with 95% Confidence Interval');
xlabel('x');
ylabel('f(x)');
grid on;

legend('95% CI','True HF','GPR Mean','HF Samples', ...
       'Location','best');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CoGPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,1,2); hold on;

fill([x; flipud(x)], ...
     [mck + 1.96*std_ck; flipud(mck - 1.96*std_ck)], ...
     [0.85 0.85 0.85], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.5);

plot(x, true, 'b', 'LineWidth', 2);
plot(x, mck, 'm-', 'LineWidth', 2);

scatter(Xc, yc, 50, '^', 'filled');
scatter(Xe, ye, 70, 'ks', 'filled');

title('CoGPR with 95% Confidence Interval');
xlabel('x');
ylabel('f(x)');
grid on;

legend('95% CI','True HF','CoGPR Mean', ...
       'LF Samples','HF Samples', ...
       'Location','best');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% multiGPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,1,3); hold on;

fill([x; flipud(x)], ...
     [mf_mean + 1.96*std_mf; flipud(mf_mean - 1.96*std_mf)], ...
     [0.85 0.85 0.85], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.5);

plot(x, true, 'b', 'LineWidth', 2);
plot(x, mf_mean, 'g:', 'LineWidth', 2);

scatter(Xc, yc, 50, '^', 'filled');
scatter(Xe, ye, 70, 'ks', 'filled');

title('multiGPR with 95% Confidence Interval');
xlabel('x');
ylabel('f(x)');
grid on;

legend('95% CI','True HF','multiGPR Mean', ...
       'LF Samples','HF Samples', ...
       'Location','best');