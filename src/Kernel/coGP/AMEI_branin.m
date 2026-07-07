clear
clc
close all

%% Branin-like multifidelity problem

f_H = @(X) ...
    (X(:,2) - (5.1/(4*pi^2))*X(:,1).^2 + 5/pi*X(:,1) - 6).^2 + ...
    10*(1-1/(8*pi))*cos(X(:,1)) + 10;

f_L = @(X) ...
    0.8*f_H(X) ...
    - 10*(X(:,1)+5) ...
    + 15;

rng(1)

Nc0 = 10;
Ne0 = 3;

lb = [-5 0];
ub = [10 15];

Xc = lhsdesign(Nc0,2);
Xc = lb + Xc.*(ub-lb);

idx = randperm(Nc0,Ne0);
Xe = Xc(idx,:);

yc = f_L(Xc);
ye = f_H(Xe);

model = multiGPR(Xc,Xe,yc,ye);

model.optimize(10);

A = [lb;
     ub];

MEPE_q = 1;

Nval = 5;

Xval = lhsdesign(Nval,2);
Xval = lb + Xval.*(ub-lb);

YL_val = f_L(Xval);
YH_val = f_H(Xval);

model.Xval = Xval;
model.yc_val = YL_val;
model.ye_val = YH_val;

n_iter = 15;

rmseHF_hist = zeros(n_iter+1,1);

pred0 = model.inference(Xval);

rmseHF_hist(1) = sqrt(mean((pred0-YH_val).^2));

for n = 1:n_iter

    fprintf('\nIteration %d\n',n);

    fid = model.chooseFID;

    if fid == 1

        fprintf('Adding LF point\n')

        x_new = find_MEPE_upd(model.model_low,A,MEPE_q,false,false);

        yL_new = f_L(x_new);

        model.Xc = [model.Xc; x_new];
        model.yc = [model.yc; yL_new];

        model.Nc = size(model.Xc,1);

    else

        fprintf('Adding HF point\n')

        x_new = find_MEPE_upd(model,A,MEPE_q,true,false);

        yL_new = f_L(x_new);
        yH_new = f_H(x_new);

        model.Xc = [model.Xc; x_new];
        model.yc = [model.yc; yL_new];

        model.Xe = [model.Xe; x_new];
        model.ye = [model.ye; yH_new];

        model.Nc = size(model.Xc,1);
        model.Ne = size(model.Xe,1);

    end
    
    x_new

    MEPE_q = MEPE_q + 1;

    model.lowreg();
    model.likelihood(model.params);
    model.optimize(5);

    predVal = model.inference(Xval);

    rmseHF_hist(n+1) = sqrt(mean((predVal-YH_val).^2));

    fprintf('Validation RMSE = %.4e\n',rmseHF_hist(n+1));

end

figure
semilogy(0:n_iter,rmseHF_hist,'-o','LineWidth',2)
grid on
xlabel('Iteration')
ylabel('Validation RMSE')
title('MEPE + AMEI')

%% Final surface plots

ngrid = 60;

[x1,x2] = meshgrid( ...
    linspace(lb(1),ub(1),ngrid), ...
    linspace(lb(2),ub(2),ngrid));

Xplot = [x1(:) x2(:)];

Ytrue = f_H(Xplot);
Ylow = f_L(Xplot);

[Ypred,Yvar] = model.inference(Xplot);

Ytrue = reshape(Ytrue,ngrid,ngrid);
Ylow  = reshape(Ylow ,ngrid,ngrid);
Ypred = reshape(Ypred,ngrid,ngrid);

Std = reshape(sqrt(diag(Yvar)),ngrid,ngrid);

figure
surf(x1,x2,Ytrue)
shading interp
title('True High Fidelity')
xlabel('x_1')
ylabel('x_2')
zlabel('f')
colorbar

figure
surf(x1,x2,Ypred)
hold on
scatter3(model.Xe(:,1),model.Xe(:,2),model.ye,...
    80,'r','filled')
shading interp
title('Final Multi-Fidelity Surrogate')
xlabel('x_1')
ylabel('x_2')
zlabel('Prediction')
colorbar

figure
surf(x1,x2,abs(Ytrue-Ypred))
shading interp
title('Absolute Prediction Error')
xlabel('x_1')
ylabel('x_2')
zlabel('|Error|')
colorbar

figure
surf(x1,x2,Std)
shading interp
title('Predictive Standard Deviation')
xlabel('x_1')
ylabel('x_2')
zlabel('\sigma')
colorbar