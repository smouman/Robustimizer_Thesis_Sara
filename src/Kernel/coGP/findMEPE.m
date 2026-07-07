function x_new = findMEPE(obj, bounds, count, plot_graphs)

if nargin < 4
        plot_graphs = false;
end

% Store info for next MEPE iteration
MEPE_q = count;

global old_Model

% "true error" 
if MEPE_q == 1
    old_Model = copy(obj);
    e_true = inf;
else
    [y_old, ~] = old_Model.inference(obj.Xe(end,:));
    % [mu_new, ~] = obj.inference(obj.Xe);
    y_new = obj.ye(end,:);

    e_true = abs(y_old - y_new);
    old_Model = copy(obj);
end

% Find CV error
e_CV = e_CV_multiGPR(obj);


% Weight factor
% alpha = MEPE_alpha_function(e_true, e_CV(end,:));
alpha = 0.5;


%% MEPE function 
fun = @(x) -( ...
    alpha * e_CV_nearest(obj, e_CV, x) + ...
    (1 - alpha) * obj.variance(x) ... 
);

lb = bounds(1,:);
ub = bounds(2,:);

x0 = lb + rand(1,obj.dim).*(ub-lb);

nvars = obj.dim;

opts = optimoptions('ga', ...
    'Display','off', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 50, ...
    'FunctionTolerance', 1e-6);

x_new = ga(fun, nvars, [], [], [], [], lb, ub, [], opts); % MEPE point found

%% OPTIONAL PLOTTING OF MEPE AND ERROR FUNCTIONS

if obj.dim == 1 && plot_graphs == 1
    xgrid = linspace(bounds(1), bounds(2), 200);

    E = zeros(size(xgrid));

    V = zeros(size(xgrid));

    T = zeros(size(xgrid));

    for i = 1:length(xgrid)
        x = xgrid(i);

        E(i) = alpha * e_CV_nearest(obj, e_CV, x) + ...
               (1 - alpha) * obj.variance(x) ...
        ;

        V(i) = obj.variance(x);

        T(i) = e_CV_nearest(obj, e_CV, x);

    end

    % Plot all together

    figure
    hold on
    
    plot(xgrid, E, 'LineWidth',2)
    plot(xgrid, V, 'LineWidth',2)
    plot(xgrid, T, 'LineWidth',2)
    
    xline(x0,'--k','Initial guess')
    
    xlabel('x')
    ylabel('Value')
    
    legend('EPE','Variance','CV Error')
    grid on

    % Plot separately

    figure;
    plot(xgrid, E, 'LineWidth', 2);
    hold on;
    xline(x0, '--k', 'Initial guess');
    title('MEPE / EPE landscape');
    xlabel('x');
    ylabel('EPE(x)');

    figure;
    plot(xgrid, V, 'LineWidth', 2);
    title('Discrepancy GP Variance');
    xlabel('x');
    ylabel('\sigma^2(x)');

    figure;
    plot(xgrid, T, 'LineWidth', 2);
    title('Discrepancy GP Variance');
    xlabel('x');
    ylabel('true error');

end


if obj.dim == 2 && plot_graphs == 2

    n = 80;

    x1 = linspace(bounds(1,1), bounds(2,1), n);
    x2 = linspace(bounds(1,2), bounds(2,2), n);

    [X1,X2] = meshgrid(x1,x2);

    Xgrid = [X1(:), X2(:)];

    E = zeros(size(Xgrid,1),1);
    V = zeros(size(Xgrid,1),1);
    T = zeros(size(Xgrid,1),1);

    for i = 1:size(Xgrid,1)

        x = Xgrid(i,:);

        V(i) = obj.variance(x);

        T(i) = e_CV_nearest(obj,e_CV,x);

        E(i) = alpha*T(i) + (1-alpha)*V(i);

    end

    E = reshape(E,n,n);
    V = reshape(V,n,n);
    T = reshape(T,n,n);

    %% --------------------------------------------------
    %% Combined figure
    %% --------------------------------------------------

    figure

    subplot(1,3,1)

    contourf(X1,X2,E,20,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),120,'r','filled')
    colorbar
    title('MEPE Criterion')

    subplot(1,3,2)

    contourf(X1,X2,V,20,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),120,'r','filled')
    colorbar
    title('Variance')

    subplot(1,3,3)

    contourf(X1,X2,T,20,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),120,'r','filled')
    colorbar
    title('Cross-validation Error')

    %% --------------------------------------------------
    %% Individual figures
    %% --------------------------------------------------

    figure

    contourf(X1,X2,E,25,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),150,'rp','filled')
    colorbar
    title('MEPE Criterion')
    xlabel('x_1')
    ylabel('x_2')

    figure

    contourf(X1,X2,V,25,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),150,'rp','filled')
    colorbar
    title('Variance')
    xlabel('x_1')
    ylabel('x_2')

    figure

    contourf(X1,X2,T,25,'LineColor','none')
    hold on
    scatter(obj.Xe(:,1),obj.Xe(:,2),60,'k','filled')
    scatter(x_new(1),x_new(2),150,'rp','filled')
    colorbar
    title('LOO Cross-validation Error')
    xlabel('x_1')
    ylabel('x_2')

end

end


function e_CV = e_CV_multiGPR(obj)

L = obj.L;

% GP residual target
r = obj.ye - obj.rho * obj.mc;

alpha = L' \ (L \ r);

invK = L' \ (L \ eye(obj.Ne));

H = invK;

e_CV = zeros(obj.Ne,1);

for i = 1:obj.Ne
    % fast LOO approximation (GP identity)
    ei = (alpha(i) / H(i,i))^2;
    e_CV(i) = ei;
end

end


function e = e_CV_nearest(obj, e_CV, x)
% k = dsearchn(obj.Xe, x);
% e = e_CV(k);

idx = knnsearch(obj.Xe, x, 'Distance', 'mahalanobis');
e = e_CV(idx);

end


function alpha = MEPE_alpha_function(e_true, e_cv_end)

if isnan(e_true) || isinf(e_true)
    alpha = 0.5;
else
    val = 0.5 * (e_true^2 / (e_cv_end + 1e-12));
    alpha = 0.99 * min(val, 1);
end

end