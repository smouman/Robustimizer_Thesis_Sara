function x_new = infillVariance(obj,bounds)

lb = bounds(1,:);
ub = bounds(2,:);

fun = @(x) -obj.variance(x);

nvars = obj.dim;

opts = optimoptions('ga', ...
    'Display','off', ...
    'PopulationSize',50, ...
    'MaxGenerations',50);

x_new = ga(fun,nvars,[],[],[],[],lb,ub,[],opts);

end