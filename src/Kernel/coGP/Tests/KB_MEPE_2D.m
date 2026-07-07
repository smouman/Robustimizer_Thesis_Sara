clear
clc
close all

% ============================================================
% 2D MULTI-FIDELITY MEPE + KRIGING BELIEVER TEST
% ============================================================

rng(1)

% High-fidelity Branin
% f_H = @(X) ...
%     (X(:,2) - 5.1/(4*pi^2).*X(:,1).^2 + 5/pi.*X(:,1) - 6).^2 + ...
%     10*(1 - 1/(8*pi)).*cos(X(:,1)) + 10;

% Low-fidelity approximation
% f_L = @(X) ...
%     0.8 * ( ...
%         (X(:,2) - 5.5/(4*pi^2).*(X(:,1)+0.3).^2 ...
%         + 4.8/pi.*X(:,1) - 5.5).^2 ...
%         + 8*(1-1/(8*pi)).*cos(0.9*X(:,1)) ...
%         + 10 ) ...
%     + 2*sin(0.5*X(:,1)).*cos(0.5*X(:,2));

% ----------------------------------------------------------
% High-fidelity Goldpr function
% ----------------------------------------------------------

% ----------------------------------------------------------
% Domain
% x1, x2 in [-2, 2]
% ----------------------------------------------------------

A = [-2 -2;
      2  2];

% ----------------------------------------------------------
% High-fidelity Goldstein-Price
% ----------------------------------------------------------

f_H = @(X) ...
( ...
    1 + ...
    (X(:,1)+X(:,2)+1).^2 .* ...
    (19 - 14*X(:,1) + 3*X(:,1).^2 ...
        - 14*X(:,2) + 6*X(:,1).*X(:,2) + 3*X(:,2).^2) ...
) ...
.* ...
( ...
    30 + ...
    (2*X(:,1)-3*X(:,2)).^2 .* ...
    (18 - 32*X(:,1) + 12*X(:,1).^2 ...
        + 48*X(:,2) - 36*X(:,1).*X(:,2) + 27*X(:,2).^2) ...
);

% ----------------------------------------------------------
% Low-fidelity model
% ----------------------------------------------------------

f_L = @(X) log10(f_H(X));

lb = A(1,:);  
ub = A(2,:);

nx = 80;

[x1,x2] = meshgrid( ...
    linspace(A(1,1),A(2,1),nx), ...
    linspace(A(1,2),A(2,2),nx));

Xtest = [x1(:),x2(:)];

y_true = f_H(Xtest);

y_low = f_L(Xtest);

% Ytrue = reshape(y_true,nx,nx);
% Ypred = reshape(y_low,nx,nx);
% 
% Ydif = Ytrue- Ypred;
% 
% figure();
% surf(x1, x2, Ydif)
% shading interp
% colorbar
% title('Model prediction')
% xlabel('x1'); ylabel('x2'); zlabel('y')
% view(135,30)
% 
% 
% figure();
% 
% surf(x1, x2, Ytrue)
% shading interp
% colorbar
% title('Model prediction')
% xlabel('x1'); ylabel('x2'); zlabel('y')
% view(135,30)
% 
% hold on;
% 
% surf(x1, x2, Ypred)
% shading interp
% colorbar
% title('Model prediction')
% xlabel('x1'); ylabel('x2'); zlabel('y')
% view(135,30)
%%

% Design space
% A = [ -5   0 ;
%       10  15 ];

lb = A(1,:);  
ub = A(2,:);  

Nc0 = 10;
Ne0 = 5;

Xc = lhsdesign(Nc0,2);
Xc = lb + Xc .* (ub - lb);

% Xe = lhsdesign(Ne0,2);
% Xe = lb + Xe .* (ub - lb);


Xe = Xc(1,:);

for i = 2:Ne0
    d = pdist2(Xc, Xe);          % distances to current HF set
    min_d = min(d,[],2);         % nearest HF distance for each point

    [~, idx] = max(min_d);       % farthest point from current HF set
    Xe = [Xe; Xc(idx,:)];
end


yc = f_L(Xc);
ye = f_H(Xe);

% Noise

noise_c = 1e-6;
noise_e = 1e-6;

% Build initial model

model = multiGPR( ...
    Xc, Xe, yc, ye, ...
    noise_c, false, ...
    noise_e, false);

model.optimize(10);
% Test grid

nx = 80;

[x1,x2] = meshgrid( ...
    linspace(A(1,1),A(2,1),nx), ...
    linspace(A(1,2),A(2,2),nx));

Xtest = [x1(:),x2(:)];

y_true = f_H(Xtest);

% Initial prediction

y_pred = model.inference(Xtest);

rmse0 = sqrt(mean((y_true-y_pred).^2));

fprintf('\nInitial RMSE = %.6f\n',rmse0)

% MEPE SETTINGS
n_iter = 5;
q      = 4;

MEPE_q = 1;

rmse_hist = zeros(n_iter+1,1);
rmse_hist(1) = rmse0;


labels = zeros(model.Ne,1);

labels(1:Ne0) = 0;   % initial design

for i = Ne0+1:model.Ne
    labels(i) = floor((i-Ne0-1)/q) + 1;
end



% MAIN LOOP
for n = 1:n_iter

    fprintf('\n====================================\n')
    fprintf('Iteration %d\n',n)
    fprintf('====================================\n')

    % Build batch with Kriging Believer

    -[X_batch, MEPE_q] = batchKB(model, A, q, MEPE_q);

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
    scatter3(model.Xe(:,1), model.Xe(:,2), model.ye, ...
    80,'r','filled');
    hold on
    
    hf_idx = (1:model.Ne)';

    labels = zeros(model.Ne,1);
    
    % mark initial HF points
    labels(1:Ne0) = 0;
    
    % everything after is batch-numbered
    for i = Ne0+1:model.Ne
        labels(i) = floor((i-Ne0-1)/q) + 1;
    end
        
    for i = 1:model.Ne
        text(model.Xe(i,1), ...
             model.Xe(i,2), ...
             model.ye(i), ...
             sprintf('%d',labels(i)), ...
             'FontSize',10, ...
             'FontWeight','bold', ...
             'HorizontalAlignment','center', ...
             'VerticalAlignment','bottom', ...
             'BackgroundColor','white', ...
             'Margin',1);
    end
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

    add_labels_2D(model.Xe, labels)

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



    %%%%%%%%%%%

%% =========================================================
%  FINAL PLOTTING (BATCH COLORED MULTI-FIDELITY RESULTS)
%% =========================================================

% ---- Batch labels ----
labels = zeros(model.Ne,1);

labels(1:Ne0) = 0;   % initial HF design

for i = Ne0+1:model.Ne
    labels(i) = floor((i-Ne0-1)/q) + 1;
end

nBatches = max(labels);
colors = lines(nBatches+1);

% ---- Grid for plotting (if not already defined) ----
nx = 80;

[x1,x2] = meshgrid( ...
    linspace(A(1,1),A(2,1),nx), ...
    linspace(A(1,2),A(2,2),nx));

Xtest = [x1(:), x2(:)];

Ytrue = f_H(Xtest);
Ylow  = f_L(Xtest);

Ypred = model.inference(Xtest);

Ytrue = reshape(Ytrue,nx,nx);
Ypred = reshape(Ypred,nx,nx);

[~,~,std] = model.inference(Xtest);
Zstd = reshape(std,nx,nx);

%% =========================================================
%  3D SURFACE PREDICTION + HF POINTS
%% =========================================================

figure
subplot(1,3,1)
hold on
grid on

surf(x1,x2,Ypred)
shading interp
view(135,30)
colorbar
title('Surrogate prediction')

for b = 0:nBatches
    idx = labels == b;

    scatter3(model.Xe(idx,1), model.Xe(idx,2), model.ye(idx), ...
        70, colors(b+1,:), 'filled');
end

%% =========================================================
%  TRUE FUNCTION
%% =========================================================

subplot(1,3,2)
surf(x1,x2,Ytrue)
shading interp
view(135,30)
colorbar
title('True function')

for b = 0:nBatches
    idx = labels == b;

    scatter3(model.Xe(idx,1), model.Xe(idx,2), model.ye(idx), ...
        40, colors(b+1,:), 'filled');
end

%% =========================================================
%  UNCERTAINTY
%% =========================================================

subplot(1,3,3)
surf(x1,x2,Zstd)
shading interp
view(135,30)
colorbar
title('Predictive std')

for b = 0:nBatches
    idx = labels == b;

    scatter3(model.Xe(idx,1), model.Xe(idx,2), model.ye(idx), ...
        40, colors(b+1,:), 'filled');
end

%% =========================================================
%  2D CONTOUR PLOTS
%% =========================================================

figure
tiledlayout(1,3)

% ---- TRUE ----
nexttile
contourf(x1,x2,Ytrue,20)
hold on
title('True function')
colorbar

for b = 0:nBatches
    idx = labels == b;
    scatter(model.Xe(idx,1),model.Xe(idx,2),100,colors(b+1,:),'filled')
end

% ---- SURROGATE ----
nexttile
contourf(x1,x2,Ypred,20)
hold on
title('Surrogate')
colorbar

for b = 0:nBatches
    idx = labels == b;
    scatter(model.Xe(idx,1),model.Xe(idx,2),100,colors(b+1,:),'filled')
end

% ---- ERROR ----
nexttile
contourf(x1,x2,abs(Ytrue-Ypred),20)
hold on
title('Absolute error')
colorbar

for b = 0:nBatches
    idx = labels == b;
    scatter(model.Xe(idx,1),model.Xe(idx,2),100,colors(b+1,:),'filled')
end

legend(arrayfun(@(b) sprintf('Batch %d',b),0:nBatches,'UniformOutput',false))








end


% RMSE history

figure

plot(0:n_iter,rmse_hist,'-o','LineWidth',2)

xlabel('Iteration')
ylabel('RMSE')
title('MEPE Convergence')

grid on

fprintf('\nFinal RMSE = %.6f\n',rmse_hist(end))





function add_labels_2D(X, labels)
    for i = 1:size(X,1)
        text(X(i,1), X(i,2), sprintf('%d',labels(i)), ...
            'FontSize',9, ...
            'FontWeight','bold', ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'BackgroundColor','white', ...
            'Margin',1);
    end
end