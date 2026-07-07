clear
clc
close all


% High fidelity (expensive truth)
f_H = @(x) (x.^2 .* sin(2*pi*x)) + 0.1*x;

% Low fidelity (biased approximation)
f_L = @(x) 0.8*(x.^2 .* sin(2*pi*x)) + 0.2*x + 0.3*sin(3*x);

rng(1);

N0 = 3;
Xc = rand(N0,1)*2 - 1;   % low fidelity points
Xe = Xc;                 % start same locations

yc = f_L(Xc);
ye = f_H(Xe);

noise_c = 1e-6;
noise_e = 1e-6;

X_mepe = [];

model = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

Xtest = linspace(-1,1,200)';
y_true = f_H(Xtest);

y_pred = model.inference(Xtest);

err_before = sqrt(mean((y_true - y_pred).^2));

fprintf("MSE before MEPE: %.6f\n", err_before);

model0 = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

A = [-1; 1];   % search domain
q = 3;


n_iter = 12;

MEPE_q = 1;

Xtest = linspace(-1,1,200)';

for t = 1:n_iter
    
    
    X_new = infillVariance(model, A);

    y_pred_before = model.inference(Xtest);

    % evaluate new points
    yL_new = f_L(X_new);
    yH_new = f_H(X_new);

    X_mepe = [X_mepe; X_new];

    model.append(X_new, yL_new, yH_new)

    % re-train
    model.lowreg();
    % model.optimize(10);

    MEPE_q = MEPE_q + 1;


    % prediction AFTER update
    y_pred_after = model.inference(Xtest);

    % =========================
    % PLOT BEFORE / AFTER
    % =========================
    figure;

    % --- BEFORE ---
    subplot(1,2,1)
    plot(Xtest, f_H(Xtest), 'k--', 'LineWidth', 1.5); hold on;
    plot(Xtest, y_pred_before, 'b', 'LineWidth', 2);
    scatter(model.Xe(1:end-1), model.ye(1:end-1), 40, 'k', 'filled');
    title(['BEFORE update t = ', num2str(t)]);
    legend('True','Pred','Data');

    % --- AFTER ---
    subplot(1,2,2)
    plot(Xtest, f_H(Xtest), 'k--', 'LineWidth', 1.5); hold on;
    plot(Xtest, y_pred_after, 'r', 'LineWidth', 2);
    scatter(model.Xe, model.ye, 40, 'k', 'filled');
    scatter(X_new, yH_new, 100, 'g', 'filled');
    title('AFTER update (new point added)');
    legend('True','Pred','Data','New point');

    drawnow;
    rmse = sqrt(mean((y_pred_after - f_H(Xtest)).^2));
    fprintf("MSE before MEPE: %.6f\n", rmse);
end






