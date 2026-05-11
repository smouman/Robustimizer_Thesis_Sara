clear; clc; close all;
rng(1);

% Test 2D function

% True function
f = @(x) sin(8*x(:,1)).*cos(8*x(:,2)) + ...
         0.5*(x(:,1)-0.5).^2 + ...
         0.25*(x(:,2)-0.5);

% Training data

Nc = 80;   
Ne = 10;   

Xc = rand(Nc,2);
Xe = rand(Ne,2);

yc = 0.8*f(Xc) + 0.5*(Xc(:,1)-0.5) - 0.3*(Xc(:,2)-0.5) + 0.2;
ye = f(Xe);

% Build model

ck = coGPR(Xc, Xe, yc, ye, 1e-6, false, 1e-6, false);

fprintf('\nTraining coGPR model...\n');
ck.optimize();
fprintf('Done.\n');

% Test data
nGrid = 60;

[x1,x2] = meshgrid(linspace(0,1,nGrid));
Xtest = [x1(:), x2(:)];

% Prediction

Ytrue = f(Xtest);
[mu, varpred] = ck.inference(Xtest);

Ztrue = reshape(Ytrue, nGrid, nGrid);
Zmu   = reshape(mu,    nGrid, nGrid);

rmse = sqrt(mean((mu - Ytrue).^2));
fprintf('\nRMSE = %.6e\n', rmse);

% Plots

figure;

subplot(1,3,1)
surf(x1,x2,Ztrue,'EdgeColor','none');
title('True Function');
xlabel('x_1'); ylabel('x_2');
view(3); colorbar;

subplot(1,3,2)
surf(x1,x2,Zmu,'EdgeColor','none');
title('coGP Prediction');
xlabel('x_1'); ylabel('x_2');
view(3); colorbar;

subplot(1,3,3)
surf(x1,x2,abs(Ztrue - Zmu),'EdgeColor','none');
title('Absolute Error');
xlabel('x_1'); ylabel('x_2');
view(3); colorbar;


% Standard deviation plot

if ~isempty(varpred)

    stdpred = sqrt(max(diag(varpred),0));
    Zstd = reshape(stdpred, nGrid, nGrid);

    figure;
    surf(x1,x2,Zstd,'EdgeColor','none');
    title('Predictive Std Dev');
    xlabel('x_1'); ylabel('x_2');
    view(3); colorbar;

end



% Confidence interval pplot

if ~isempty(varpred)

    stdpred = sqrt(max(diag(varpred),0));
    Zstd = reshape(stdpred, nGrid, nGrid);

    Zmu = reshape(mu, nGrid, nGrid);

    Zupper = Zmu + 1.96 * Zstd;
    Zlower = Zmu - 1.96 * Zstd;

    figure;

    surf(x1,x2,Zmu,'EdgeColor','none');
    hold on;

    % Upper bound
    surf(x1,x2,Zupper,'EdgeColor','none','FaceAlpha',0.3);

    % Lower bound
    surf(x1,x2,Zlower,'EdgeColor','none','FaceAlpha',0.3);

    title('Prediction with 95% Confidence Interval');
    xlabel('x_1'); ylabel('x_2');
    view(3);
    colorbar;

end