clear
clc
close all

rng(1);


f_H = @(x) ( ...
    x(:,1).^2 .* sin(2*pi*x(:,1)) + ...
    x(:,2).^2 .* cos(2*pi*x(:,2)) ...
    + 0.1*(x(:,1)+x(:,2)) );

f_L = @(x) ( ...
    0.8*(x(:,1).^2 .* sin(2*pi*x(:,1)) + ...
         x(:,2).^2 .* cos(2*pi*x(:,2))) ...
    + 0.2*(x(:,1)+x(:,2)) ...
    + 0.3*sin(3*x(:,1)) );



% =========================
% 2D initial design
% =========================

Nc0 = 10;   % low-fidelity samples
Ne0 = 5;    % high-fidelity samples

Xc = rand(Nc0,2)*2 - 1;

idx = randperm(Nc0, Ne0);
Xe = Xc(idx,:);

yc = f_L(Xc);
ye = f_H(Xe);

noise_c = 1e-6;
noise_e = 1e-6;

model = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

% =========================
% test grid (for visualization)
% =========================

nGrid = 40;
[x1,x2] = meshgrid(linspace(-1,1,nGrid), linspace(-1,1,nGrid));
Xtest = [x1(:), x2(:)];

y_true = f_H(Xtest);

y_pred = model.inference(Xtest);

err_before = sqrt(mean((y_true - y_pred).^2));
fprintf("RMSE before MEPE: %.6f\n", err_before);

% =========================
% MEPE loop
% =========================
n_iter = 30;
MEPE_q = 1;

for t = 1:n_iter

    % ---- MEPE proposal ----
    X_new = findMEPE(model, [-1 -1; 1 1], MEPE_q);

    y_pred_before = model.inference(Xtest);

    % ---- evaluate new point ----
    yL_new = f_L(X_new);
    yH_new = f_H(X_new);

    model.append(X_new, yL_new, yH_new)

    model.lowreg();
    model.likelihood(model.params);
    % model.optimize(5);

    MEPE_q = MEPE_q + 1;

    % ---- prediction after ----
    y_pred_after = model.inference(Xtest);

    % =========================
    % PLOT (2D CONTOURS)
    % =========================
    figure;

    % --- BEFORE ---
    subplot(1,2,1)
    contourf(x1, x2, reshape(f_H(Xtest), nGrid, nGrid), 20, 'LineStyle','none');
    colorbar; hold on;
    scatter(model.Xe(1:end-1,1), model.Xe(1:end-1,2), 40, 'k', 'filled');
    title(['BEFORE t = ', num2str(t)]);

    % --- AFTER ---
    subplot(1,2,2)
    contourf(x1, x2, reshape(y_pred_after, nGrid, nGrid), 20, 'LineStyle','none');
    colorbar; hold on;
    scatter(model.Xe(:,1), model.Xe(:,2), 40, 'k', 'filled');
    scatter(X_new(:,1), X_new(:,2), 120, 'g', 'filled');
    title('AFTER update');

    drawnow;

    % RMSE
    rmse = sqrt(mean((y_pred_after - y_true).^2));
    disp(rmse);

end