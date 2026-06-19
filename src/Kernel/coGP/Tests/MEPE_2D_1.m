% Test MEPE criterion in 2D case

clear
clc
close all

rng(1)

%% High-fidelity Branin

f_H = @(X) ...
    (X(:,2) - 5.1/(4*pi^2).*X(:,1).^2 + 5/pi.*X(:,1) - 6).^2 + ...
    10*(1 - 1/(8*pi)).*cos(X(:,1)) + 10;

%% Low-fidelity approximation

f_L = @(X) ...
    0.8*f_H(X) ...
    - 10*(0.1*X(:,1) + 0.05*X(:,2)) ...
    - 2;

%% Design space
A = [ -5   0 ;
      10  15 ];

lb = A(1,:);  
ub = A(2,:);  

Nc0 = 10;
Ne0 = 5;

Xc = lhsdesign(Nc0,2);
Xc = lb + Xc .* (ub - lb);

Xe = lhsdesign(Ne0,2);
Xe = lb + Xe .* (ub - lb);



% --- pick first HF point randomly ---
Xe = Xc(1,:);

for i = 2:Ne0
    d = pdist2(Xc, Xe);          % distances to current HF set
    min_d = min(d,[],2);         % nearest HF distance for each point

    [~, idx] = max(min_d);       % farthest point from current HF set
    Xe = [Xe; Xc(idx,:)];
end


yc = f_L(Xc);
ye = f_H(Xe);

noise_c = 1e-6;
noise_e = 1e-6;

model = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

% =========================
% test grid (for visualization)
% =========================
nx = 80;

[x1,x2] = meshgrid( ...
    linspace(A(1,1),A(2,1),nx), ...
    linspace(A(1,2),A(2,2),nx));

Xtest = [x1(:),x2(:)];

y_true = f_H(Xtest);

y_pred = model.inference(Xtest);

err_before = sqrt(mean((y_true - y_pred).^2));
fprintf("RMSE before MEPE: %.6f\n", err_before);

% =========================
% MEPE loop
% =========================
n_iter = 20;
MEPE_q = 1;

for t = 1:n_iter

    % ---- MEPE proposal ----
    X_new = find_MEPE(model, A, MEPE_q);

    y_pred_before = model.inference(Xtest);

    % ---- evaluate new point ----
    yL_new = f_L(X_new);
    yH_new = f_H(X_new);

    model.append(X_new, yL_new, yH_new)

    model.lowreg();
    model.optimize();

    MEPE_q = MEPE_q + 1;

    % ---- prediction after ----
    y_pred_after = model.inference(Xtest);

    y_true = f_H(Xtest);

    % =========================
    % PLOT (2D CONTOURS)
    % =========================
    figure;

    % --- BEFORE ---
    subplot(1,2,1)
    contourf(x1, x2, reshape(y_true, nx, nx), 20, 'LineStyle','none');
    colorbar; hold on;
    scatter(model.Xe(1:end-1,1), model.Xe(1:end-1,2), 40, 'k', 'filled');
    title('True surrogate model');

    % --- AFTER ---
    subplot(1,2,2)
    contourf(x1, x2, reshape(y_pred_after, nx, nx), 20, 'LineStyle','none');
    colorbar; hold on;
    scatter(model.Xe(:,1), model.Xe(:,2), 40, 'k', 'filled');
    scatter(X_new(:,1), X_new(:,2), 120, 'g', 'filled');
    title(['AFTER update', num2str(t)]);

    drawnow;

    % RMSE
    rmse = sqrt(mean((y_pred_after - y_true).^2));
    disp(rmse);

end