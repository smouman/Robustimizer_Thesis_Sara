clear
clc
close all

%% ============================================================
% 2D MULTI-FIDELITY MEPE + KRIGING BELIEVER TEST
%% ============================================================

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


Xe = Xc(1,:);

for i = 2:Ne0
    d = pdist2(Xc, Xe);          % distances to current HF set
    min_d = min(d,[],2);         % nearest HF distance for each point

    [~, idx] = max(min_d);       % farthest point from current HF set
    Xe = [Xe; Xc(idx,:)];
end


yc = f_L(Xc);
ye = f_H(Xe);

%% Noise

noise_c = 1e-6;
noise_e = 1e-6;

%% Build initial model

model = multiGPR( ...
    Xc, Xe, yc, ye, ...
    noise_c, false, ...
    noise_e, false);

model.optimize(10);
%% Test grid

nx = 80;

[x1,x2] = meshgrid( ...
    linspace(A(1,1),A(2,1),nx), ...
    linspace(A(1,2),A(2,2),nx));

Xtest = [x1(:),x2(:)];

y_true = f_H(Xtest);

%% Initial prediction

y_pred = model.inference(Xtest);

rmse0 = sqrt(mean((y_true-y_pred).^2));

fprintf('\nInitial RMSE = %.6f\n',rmse0)

% MEPE SETTINGS
n_iter = 5;
q      = 3;

MEPE_q = 1;

rmse_hist = zeros(n_iter+1,1);
rmse_hist(1) = rmse0;


% MAIN LOOP
for n = 1:n_iter

    fprintf('\n====================================\n')
    fprintf('Iteration %d\n',n)
    fprintf('====================================\n')

    % Build batch with Kriging Believer

    [X_batch, MEPE_q] = batchKB(model, A, q, MEPE_q);

    % Evaluate expensive model
    for i = 1:q
        X_new = X_batch(i,:);
        yL_new = f_L(X_new);
        yH_new = f_H(X_new);
        model.append(X_new, yL_new, yH_new)
    end

    % Retrain surrogate

    model.lowreg();
    model.optimize(10);

    % Evaluate surrogate

    y_pred = model.inference(Xtest);

    rmse = sqrt(mean((y_true-y_pred).^2));

    rmse_hist(n+1) = rmse;

    fprintf('RMSE = %.6f\n',rmse)

    % Visualization

    Ytrue = reshape(y_true,nx,nx);
    Ypred = reshape(y_pred,nx,nx);
    
    figure();
    subplot(1,3,1)
    surf(x1, x2, Ypred)
    shading interp
    colorbar
    title('Model prediction')
    xlabel('x1'); ylabel('x2'); zlabel('y')
    view(135,30)

    hold on
    
    scatter3(model.Xe(:,1), model.Xe(:,2), model.ye, ...
             50, 'k', 'filled')
    subplot(1,3,2)
    surf(x1, x2, Ytrue)
    shading interp
    colorbar
    title('True function')
    view(135,30)

    subplot(1,3,3)
    [mu, ~, std] = model.inference(Xtest);
    Zstd = reshape(std, size(x1));
    surf(x1, x2, Zstd)
    shading interp
    xlabel('x1')
    ylabel('x2')
    zlabel('Std deviation')
    title('GP Uncertainty Surface')
    colorbar

    figure
    contourf(x1, x2, Zstd, 20)
    colorbar
    xlabel('x1')
    ylabel('x2')
    title('GP Uncertainty (Std Dev)')

    %% TRUE

    figure();
    subplot(1,3,1)

    contourf(x1,x2,Ytrue,20)
    hold on

    scatter( ...
        model.Xe(:,1), ...
        model.Xe(:,2), ...
        40,'k','filled')

    scatter( ...
        X_batch(:,1), ...
        X_batch(:,2), ...
        100,'r','filled')

    title('True Function')

    colorbar

    %% PREDICTED

    subplot(1,3,2)

    contourf(x1,x2,Ypred,20)
    hold on

    scatter( ...
        model.Xe(:,1), ...
        model.Xe(:,2), ...
        40,'k','filled')

    scatter( ...
        X_batch(:,1), ...
        X_batch(:,2), ...
        100,'r','filled')

    title(sprintf('Surrogate (%d samples)',model.Ne))

    colorbar

    %% ERROR

    subplot(1,3,3)

    contourf(x1,x2,abs(Ytrue-Ypred),20)
    hold on

    scatter( ...
        model.Xe(:,1), ...
        model.Xe(:,2), ...
        40,'k','filled')

    scatter( ...
        X_batch(:,1), ...
        X_batch(:,2), ...
        100,'r','filled')

    title('Absolute Error')

    colorbar

    drawnow

end


% RMSE history

figure

plot(0:n_iter,rmse_hist,'-o','LineWidth',2)

xlabel('Iteration')
ylabel('RMSE')
title('MEPE Convergence')

grid on

fprintf('\nFinal RMSE = %.6f\n',rmse_hist(end))