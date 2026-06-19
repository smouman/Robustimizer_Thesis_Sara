clear
clc
close all


% High fidelity (expensive truth)
f_H = @(x) (x.^2 .* sin(2*pi*x)) + 0.1*x;

% Low fidelity (biased approximation)
f_L = @(x) 0.8*(x.^2 .* sin(2*pi*x)) + 0.2*x + 0.3*sin(3*x);

rng(1);

Nc0 = 10;   % low-fidelity samples
Ne0 = 3;    % high-fidelity samples

Xc = rand(Nc0,1)*2 - 1;

idx = randperm(Nc0, Ne0);
Xe = Xc(idx,:);


yc = f_L(Xc);
ye = f_H(Xe);

noise_c = 1e-6;
noise_e = 1e-6;

X_mepe = [];

model = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

Xtest = linspace(-1,1,200)';
y_true = f_H(Xtest);

y_pred = model.inference(Xtest);

rmse0 = sqrt(mean((y_true - y_pred).^2));

fprintf("MSE before MEPE: %.6f\n", rmse0);

model0 = multiGPR(Xc, Xe, yc, ye, noise_c, false, noise_e, false);

A = [-1; 1];   % search domain
q = 3;


n_iter = 12;

MEPE_q = 1;

Xtest = linspace(-1,1,200)';

rmse_hist = zeros(n_iter+1,1);
rmse_hist(1) = rmse0;

for t = 1:n_iter
    
    
    X_new = findMEPE(model, A, MEPE_q);

    y_pred_before = model.inference(Xtest);

    % evaluate new points
    yL_new = f_L(X_new);
    yH_new = f_H(X_new);

    X_mepe = [X_mepe; X_new];

    model.append(X_new, yL_new, yH_new)

    % re-train
    model.lowreg();

    % model.optimize();

    MEPE_q = MEPE_q + 1;


    % prediction AFTER update
    [y_pred_after,~,std_after] = model.inference(Xtest);

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

    upper = y_pred_after + 1.96*std_after;
    lower = y_pred_after - 1.96*std_after;
    fill([Xtest; flipud(Xtest)], ...
     [upper; flipud(lower)], ...
     [0.8 0.8 1], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.3);

    hold on
 

    % --- ADD LABELS ---
    for i = 1:size(model.Xe,1)
        text(model.Xe(i), model.ye(i), sprintf('%d', i), ...
            'VerticalAlignment','bottom', ...
            'HorizontalAlignment','right', ...
            'FontSize',12, ...
            'Color','blue');
    end

    title('AFTER update (new point added)');
    legend('True','Pred','Data','New point');

    drawnow;
    rmse = (sqrt(mean((y_pred_after - f_H(Xtest)).^2)));
    rmse_hist(t+1) = rmse;
end

figure

plot(0:n_iter,rmse_hist,'-o','LineWidth',2)

xlabel('Iteration')
ylabel('RMSE')
title('MEPE Convergence')

grid on

fprintf('\nFinal RMSE = %.6f\n',rmse_hist(end))





