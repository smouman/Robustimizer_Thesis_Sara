%% Initial design
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

model = multiGPR(Xc,Xe,yc,ye);

model.optimize(10);

A = [-1; 1];

MEPE_q = 1;

%% Fixed validation set (same idea as AMEI paper)

Nval = 5;

Xval = lhsdesign(Nval,1);
Xval = -1 + Xval.*(2);

YL_val = f_L(Xval);
YH_val = f_H(Xval);

n_iter = 12;
%% History

rmseHF_hist = zeros(n_iter+1,1);

[pred0, ~, ~] = model.inference(Xval);

rmseHF_hist(1) = sqrt(mean((pred0-YH_val).^2));

model.Xval = Xval;
model.yc_val = YL_val;
model.ye_val = YH_val;


for n = 1:n_iter

    fprintf('\nIteration %d\n',n);     

    % Fidelity selection

    fid = model.chooseFID;

    % fid = 2;

    if fid == 1

        %% LOW-FIDELITY UPDATE

        x_new = find_MEPE_upd(model.model_low,A, MEPE_q, false, true);

        fprintf('Adding LF point\n');

        yL_new = f_L(x_new);

        model.Xc = [model.Xc; x_new];
        model.yc = [model.yc; yL_new];

        model.Nc = size(model.Xc,1);

    else
        
        %% HIGH-FIDELITY UPDATE

        fprintf('Adding HF point\n');

        x_new = find_MEPE_upd(model,A, MEPE_q, true, true);

        yL_new = f_L(x_new);
        yH_new = f_H(x_new);

        model.Xc = [model.Xc; x_new];
        model.yc = [model.yc; yL_new];

        model.Xe = [model.Xe; x_new];
        model.ye = [model.ye; yH_new];

        model.Nc = size(model.Xc,1);
        model.Ne = size(model.Xe,1);

    end

    %% Rebuild MF model
    MEPE_q = MEPE_q + 1;
    model.lowreg();
    model.optimize(5);

    %% Validation RMSE

    predVal = model.inference(Xval);

    rmseHF_hist(n+1) = ...
        sqrt(mean((predVal - YH_val).^2));

    fprintf('Validation RMSE = %.4e\n', ...
        rmseHF_hist(n+1));

end

figure;

semilogy(0:n_iter,rmseHF_hist,'-o','LineWidth',2);

xlabel('Iteration');
ylabel('HF Validation RMSE');
title('MEPE + AMEI Fidelity Selection');

grid on;



%% Plot final surrogate

Xplot = linspace(-1,1,300)';

Ytrue = f_H(Xplot);
Ylow  = f_L(Xplot);

[Ypred, Yvar] = model.inference(Xplot);

figure;
hold on;
grid on;

% True HF function
plot(Xplot, Ytrue,'k-','LineWidth',2);

% Final surrogate
plot(Xplot, Ypred,'b-','LineWidth',2);

% Low-fidelity function (optional)
plot(Xplot, Ylow,'--','Color',[0.6 0.6 0.6],'LineWidth',1.5);

% HF samples
scatter(model.Xe,model.ye,70,'r','filled');

% LF-only samples
idxLF = setdiff(1:model.Nc,1:model.Ne);
scatter(model.Xc(idxLF),model.yc(idxLF),40,'ko');

legend('True HF','MF surrogate','Low-fidelity','HF samples','LF samples',...
       'Location','best');

xlabel('x');
ylabel('y');
title('Final Multi-Fidelity Surrogate');