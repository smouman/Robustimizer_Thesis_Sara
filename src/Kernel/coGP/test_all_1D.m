clear all
close all
clc

rng(10)

hf = @(x) (6*x - 2).^2 .* sin(12*x - 4);

A = 0.5; B = 10; C = 5;
lf = @(x) A*hf(x) + B*(x - 0.5) - C;


Xe = [0; 0.4; 0.6; 1];
ye = hf(Xe);

Xc = linspace(0,1,11)';
yc = lf(Xc);

x = linspace(0,1,100)';

figure(1); clf
plot(x, lf(x), 'r', 'LineWidth', 1.5); hold on
plot(x, hf(x), 'b', 'LineWidth', 2)

scatter(Xc, yc, 60, '^', 'filled')
scatter(Xe, ye, 80, 'bs', 'filled')

% Regression + correlation functions
regr = @regpoly2;     % constant trend
corr = @corrgauss;    % Gaussian kernel

theta0 = ones(1,1);
lob = 1e-6 * ones(1,1);
upb = 10000   * ones(1,1);

[dmodel, ~] = dacefit(Xe, ye, regr, corr, theta0, lob, upb);

[mdace, ~] = predictor(x, dmodel);

rmse_dace = sqrt(mean((mdace - true).^2));

plot(x, mdace, 'm-', 'LineWidth', 2)

% GP

gpr = GPR(Xe, ye, [], true);
gpr.optimize();

[mg, vg] = gpr.inference(x);

plot(x, mg, 'k--', 'LineWidth', 2)

% Co-kriging

ck = coGPR(Xc, Xe, yc, ye);
ck.optimize();

[mck, vck] = ck.inference(x);

plot(x, mck, 'm-', 'LineWidth', 2)

disp(ck.params);

% Recursive MFGP

mf = multiGPR(Xc, Xe, yc, ye);
mf.optimize();

[mf_mean, mf_var] = mf.inference(x);

% Dace




% Plot



plot(x, mf_mean, 'g:', 'LineWidth', 2)

legend( ...
    'Low fidelity', ...
    'High fidelity', ...
    'Low samples', ...
    'High samples', ...
    'Dace', ...
    'GPR', ...
    'CoGPR', ...
    'multiGPR', ...
    'Location', 'SouthWest')

title('Forrester Multi-Fidelity Comparison')
xlabel('x')
ylabel('f(x)')
grid on

% Error metrics

true = hf(x);

fprintf('\n==============================\n');
fprintf('RMSE RESULTS\n');
fprintf('==============================\n');

fprintf('DACE      : %.6e\n', sqrt(mean((mdace - true).^2)));
fprintf('GPR       : %.6f\n', sqrt(mean((mg - true).^2)));
fprintf('CoGPR     : %.6f\n', sqrt(mean((mck - true).^2)));
fprintf('multiGPR  : %.6f\n', sqrt(mean((mf_mean - true).^2)));
