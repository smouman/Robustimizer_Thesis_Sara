classdef GPR < handle
    properties
        % Data
        X
        y
        N
        dim

        % Noise
        noise
        noise_fix

        % Stability
        stab = 1e-6

        % Hyperparameters
        params
        bound
        id_theta

        theta

        % GP quantities
        K
        L
        alpha
        NLML
    end

    methods

        % ======================================================
        % Constructor
        % ======================================================
        function obj = GPR(X, y, noise_var, noise_fix)

            if nargin < 3, noise_var = []; end
            if nargin < 4, noise_fix = false; end

            obj.X = X;
            obj.y = y;

            obj.N = size(X,1);
            obj.dim = size(X,2);

            obj.noise = noise_var;
            obj.noise_fix = noise_fix;

            obj.bound = {};
            obj.params = obj.hyperparams();

            obj.likelihood(obj.params);
        end

        % ======================================================
        % Hyperparameters
        % ======================================================
        function hyper = hyperparams(obj)

            hyper = ones(obj.dim+1,1);

            obj.id_theta = 1:length(hyper);

            for i = 1:(obj.dim+1)
                obj.bound{end+1} = [1e-6 Inf];
            end

            if ~isempty(obj.noise) && ~obj.noise_fix
                hyper = [hyper; obj.noise];
                obj.bound{end+1} = [1e-6 Inf];
            end
        end

        % ======================================================
        % RBF Kernel (standard form)
        % ======================================================
        function K = RBF(obj, hyper, X1, X2)

            if nargin < 4
                X2 = X1;
            end
        
            sigma_f = hyper(1);
            len_sc = hyper(2:end)';
        
            % scale inputs
            X1s = X1 ./ len_sc;
            X2s = X2 ./ len_sc;
        
            % pairwise squared distances
            K = zeros(size(X1,1), size(X2,1));
        
            for i = 1:size(X2,1)
                diff = X1s - X2s(i,:);
                K(:,i) = sum(diff.^2,2);
            end
        
            K = sigma_f * exp(-0.5 * K);
        end

        % ======================================================
        % Log Marginal Likelihood (NLML)
        % ======================================================
        function NLML = likelihood(obj, hyper)

            if ~isempty(obj.noise) && ~obj.noise_fix
                sigma_n = hyper(end);
            else
                sigma_n = 0;
            end

            theta = hyper(obj.id_theta);
            obj.theta = theta;

            K = obj.RBF(theta, obj.X) + eye(obj.N)*sigma_n;
            obj.K = K;

            try
                L = chol(K + eye(obj.N)*obj.stab,'lower');
            catch
                NLML = 1e10;
                return;
            end

            obj.L = L;

            alpha = L'\(L\obj.y);
            obj.alpha = alpha;

            NLML = 0.5*obj.y'*alpha + ...
                   sum(log(diag(L))) + ...
                   0.5*obj.N*log(2*pi);

            obj.NLML = NLML;
        end

        % ======================================================
        % Optimization (multi-start compatible)
        % ======================================================
        function optimize(obj, restart)

            if nargin < 2, restart = 10; end

            objFun = @(p) obj.likelihood(p);

            lb = cellfun(@(b)b(1), obj.bound)';
            ub = cellfun(@(b)b(2), obj.bound)';

            opts = optimoptions('fmincon', ...
                'Algorithm','interior-point', ...
                'Display','off', ...
                'MaxIterations',300);

            bestObj = Inf;
            bestP = obj.params;

            for k = 1:(restart+1)

                if k == 1
                    p0 = obj.params;
                else
                    p0 = 0.5 + rand(size(obj.params));
                end

                try
                    [p,fval] = fmincon(objFun, p0, [],[],[],[], lb, ub, [], opts);

                    if fval < bestObj
                        bestObj = fval;
                        bestP = p;
                    end
                catch
                end
            end

            obj.params = bestP;
            obj.likelihood(obj.params);
        end

        % ======================================================
        % Inference
        % ======================================================
        function [mean,var,std] = inference(obj, x, return_std)

            k_s  = obj.RBF(obj.theta, x, obj.X);
            k_ss = obj.RBF(obj.theta, x, x);

            alpha = obj.L'\(obj.L\obj.y);

            mean = k_s * alpha;

            v = obj.L\k_s';
            var = k_ss - v'*v;

            std = sqrt(max(diag(var),0));

            if nargin < 4 || ~return_std
                std = [];
            end
        end


        function [var] = variance(obj, x)

            k_s  = obj.RBF(obj.theta, x, obj.X);
            k_ss = obj.RBF(obj.theta, x, x);

            alpha = obj.L'\(obj.L\obj.y);

            mean = k_s * alpha;

            v = obj.L\k_s';
            var = k_ss - v'*v;

        end

    end
end