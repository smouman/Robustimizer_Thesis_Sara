% Test performance of GPR, coGPR, multiGPR in 2D case, with timing


clear; close all; clc;
rng(10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D TEST FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hf = @(x) ...
    sin(8*x(:,1)).*cos(8*x(:,2)) + ...
    0.5*(x(:,1)-0.5).^2 + ...
    0.25*(x(:,2)-0.5);

A = 0.5; B = 10; C = 5;
lf = @(x) A*hf(x) + B*(x(:,1)-0.5) - C*(x(:,2)-0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAINING DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Nc = 40;
Ne = 10;

% Cheap points
Xc = rand(Nc,2);

% Select expensive points from cheap points
idx = randperm(Nc, Ne);
Xe = Xc(idx,:);

% Evaluate models
yc = lf(Xc);
ye = hf(Xe);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST GRID
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nGrid = 40;

[x1,x2] = meshgrid(linspace(0,1,nGrid), linspace(0,1,nGrid));
Xtest = [x1(:), x2(:)];

true = hf(Xtest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 1. GPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_gpr_build = timeit(@() GPR(Xe, ye, [], true));

gpr = GPR(Xe, ye, [], true);

t_gpr_opt = timeit(@() gpr.optimize());
t_gpr_inf = timeit(@() gpr.inference(Xtest));

[mg, ~] = gpr.inference(Xtest);

rmse_gpr = sqrt(mean((mg - true).^2)) / (max(true) - min(true));


err = abs(mg - true);

fprintf('RMSE      : %.6f\n', sqrt(mean(err.^2)));
fprintf('MAX ERROR : %.6f\n', max(err));
fprintf('MEAN ERROR: %.6f\n', mean(err));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 2. CoGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_ck_build = timeit(@() coGPR(Xc, Xe, yc, ye));

ck = coGPR(Xc, Xe, yc, ye);

t_ck_opt = timeit(@() ck.optimize());
t_ck_inf = timeit(@() ck.inference(Xtest));

[mck, ~] = ck.inference(Xtest);

rmse_ck = sqrt(mean((mck - true).^2)) / (max(true) - min(true));

err = abs(mck - true);

fprintf('RMSE      : %.6f\n', sqrt(mean(err.^2)));
fprintf('MAX ERROR : %.6f\n', max(err));
fprintf('MEAN ERROR: %.6f\n', mean(err));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== 3. multiGPR =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_mf_build = timeit(@() multiGPR(Xc, Xe, yc, ye));

mf = multiGPR(Xc, Xe, yc, ye);

t_mf_opt = timeit(@() mf.optimize());
t_mf_inf = timeit(@() mf.inference(Xtest));

[mf_mean, ~] = mf.inference(Xtest);

rmse_mf = sqrt(mean((mf_mean - true).^2)) / (max(true) - min(true));

err = abs(mf_mean - true);

fprintf('RMSE      : %.6f\n', sqrt(mean(err.^2)));
fprintf('MAX ERROR : %.6f\n', max(err));
fprintf('MEAN ERROR: %.6f\n', mean(err));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RESHAPE FOR PLOTTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ztrue = reshape(true, nGrid, nGrid);
Zgpr  = reshape(mg, nGrid, nGrid);
Zck   = reshape(mck, nGrid, nGrid);
Zmf   = reshape(mf_mean, nGrid, nGrid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== PLOTS =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

subplot(2,2,1)
surf(x1,x2,Ztrue,'EdgeColor','none');
title('True Function');
xlabel('x1'); ylabel('x2');
view(3); colorbar;

subplot(2,2,2)
surf(x1,x2,Zgpr,'EdgeColor','none');
title('GPR');
xlabel('x1'); ylabel('x2');
view(3); colorbar;

subplot(2,2,3)
surf(x1,x2,Zck,'EdgeColor','none');
title('CoGPR');
xlabel('x1'); ylabel('x2');
view(3); colorbar;

subplot(2,2,4)
surf(x1,x2,Zmf,'EdgeColor','none');
title('multiGPR');
xlabel('x1'); ylabel('x2');
view(3); colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===================== METRICS =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n====================================\n');
fprintf('2D RMSE RESULTS\n');
fprintf('====================================\n');

fprintf('GPR       : %.6e\n', rmse_gpr);
fprintf('CoGPR     : %.6e\n', rmse_ck);
fprintf('multiGPR  : %.6e\n', rmse_mf);

fprintf('\n====================================\n');
fprintf('MODEL BUILD TIME (s)\n');
fprintf('====================================\n');

fprintf('GPR       : %.4f\n', t_gpr_build);
fprintf('CoGPR     : %.4f\n', t_ck_build);
fprintf('multiGPR  : %.4f\n', t_mf_build);

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