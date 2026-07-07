classdef multiGPR < matlab.mixin.Copyable
    properties
        Xc
        Xe
        Xval

        yc
        ye
        ye_val
        yc_val

        Nc
        Ne
        N
        dim

        noise_c
        noise_e
        noise_fix_c
        noise_fix_e

        stab = 1e-10

        model_low
        mc
        covc

        params
        bound
        id_theta_e

        theta_e
        rho

        alpha
        NLML

        L 
    end

    methods

        function obj = multiGPR(Xc, Xe, yc, ye, noise_var_c, noise_fix_c, noise_var_e, noise_fix_e)

            if nargin < 5, noise_var_c = []; end
            if nargin < 6, noise_fix_c = false; end
            if nargin < 7, noise_var_e = []; end
            if nargin < 8, noise_fix_e = false; end

            obj.Xc = Xc;
            obj.Xe = Xe;
            obj.yc = yc;
            obj.ye = ye;

            obj.Nc = size(Xc,1);
            obj.Ne = size(Xe,1);
            obj.N  = obj.Nc + obj.Ne;
            obj.dim = size(Xc,2);

            obj.noise_c = noise_var_c;
            obj.noise_e = noise_var_e;
            obj.noise_fix_c = noise_fix_c;
            obj.noise_fix_e = noise_fix_e;

            obj.bound = {};
            obj.params = obj.hyperparams();

            obj.lowreg();
            obj.likelihood(obj.params);
        end


        function hyper = hyperparams(obj)

            hyper_e = ones(obj.dim+1,1) + 0.1;
        
            obj.id_theta_e = 1:length(hyper_e);
        
            for i = 1:(obj.dim+1)
                obj.bound{end+1} = [1e-6 Inf];
            end
        
            % noise (optional)
            if ~isempty(obj.noise_e) && ~obj.noise_fix_e
                hyper_e = [hyper_e; obj.noise_e];
                obj.bound{end+1} = [1e-6 Inf];
            end

            hyper = hyper_e;

        end

        % Low-fidelity model
        function lowreg(obj)
            obj.model_low = GPR(obj.Xc, obj.yc, obj.noise_c, obj.noise_fix_c);
            obj.model_low.optimize(10);

            [obj.mc, obj.covc] = obj.model_low.inference(obj.Xe);
        end

        % RBF covariance matrix
        function K = RBF(obj,hyper,X1,X2)

            if nargin < 4
                X2 = X1;
            end

            sf = hyper(1);      %signal variance
            len_sc = hyper(2:end); %length scale


            r = permute((X1 ./ len_sc'),[1 3 2]) - permute(X2 ./ len_sc', [3 1 2]);

            K = sf * exp(-0.5 * sum(r.^2,3));
        end


        function [NLML, grad] = likelihood(obj, hyper)

            d = obj.dim;

            sf_e  = hyper(1);
            len_sc_e = hyper(2:d+1);
        
            idx = d + 2;
        
            if ~isempty(obj.noise_e) && ~obj.noise_fix_e
                sigma_ne = hyper(idx);
            else
                sigma_ne = 0;
            end
        
            obj.theta_e = [sf_e; len_sc_e(:)];    
            
            % kernels
            K = obj.RBF(obj.theta_e, obj.Xe) + eye(obj.Ne)*(sigma_ne^2);
            L = chol(K + eye(obj.Ne)*obj.stab,'lower');

            obj.L = L;

            alpha1 = L'\(L\obj.mc);
            alpha2 = L'\(L\obj.ye);

            obj.rho = (obj.mc' * alpha2) / (obj.mc' * alpha1);

            obj.alpha = L'\(L\(obj.ye - obj.rho*obj.mc));

            r = obj.ye - obj.rho*obj.mc;

            NLML = sum(log(diag(L))) + ...
                   0.5*(r' * obj.alpha) + ...
                   0.5*obj.Ne*log(2*pi);

            obj.NLML = NLML;
        
            % Gradient calculation

            if nargout > 1
        
                invK = L'\(L\eye(obj.Ne));
        
                Q = invK - obj.alpha*obj.alpha';
        
                grad = zeros(length(hyper),1);
        
                %derivative wrt sf and len_sc
                for i = 1:length(obj.theta_e)
        
                    dK = obj.kernel_derivative(obj.Xe, obj.theta_e, i);
        
                    grad(i) = 0.5 * trace(Q * dK);
                end
        
                % noise gradient
                if ~isempty(obj.noise_e) && ~obj.noise_fix_e
                    dK_noise = 2*sigma_ne*eye(obj.Ne);
                    grad(end) = 0.5 * trace(Q * dK_noise);
                end
            end
        end


        function dK = kernel_derivative(obj, X, hyper, i)

            sf = hyper(1);
            len_sc = hyper(2:end);
        
            K = obj.RBF(hyper, X);
        
            if i == 1
                % derivative wrt signal variance
                dK = K / sf;
        
            else
                % derivative wrt lengthscale i-1
                j = i-1;
        
                Xi = X(:,j);
        
                D = (Xi - Xi').^2;
        
                dK = K .* D / (len_sc(j)^3);
            end
        end


        function optimize(obj, restart)

            if nargin < 2, restart = 10; end
            
            lb = cellfun(@(b)b(1), obj.bound)';
            ub = cellfun(@(b)b(2), obj.bound)';

            opts = optimoptions('fmincon', ...
                'Algorithm','interior-point', ...
                'Display','off', ...
                'SpecifyObjectiveGradient',true, ...
                'CheckGradients',false, ...
                'FiniteDifferenceType','central', ...
                'MaxIterations',500, ...
                'MaxFunctionEvaluations',1e5, ...
                'OptimalityTolerance',1e-10, ...
                'StepTolerance',1e-10, ...
                'ConstraintTolerance',1e-10);
            
            best = Inf;
            bestP = obj.params;
            
            % Very sensitivite to scale!!!
            scale = 1;
            
            for k = 1:(restart+1)
            
                if k == 1
                    p0 = obj.params;
                else
                    p0 = exp(scale*randn(size(obj.params))) .* obj.params;
                    p0 = max(min(p0,ub),lb);
                end

                objFun = @(p) obj.likelihood(p);
            
                try
                    [p,fval] = fmincon(objFun, p0, [],[],[],[], lb, ub, [], opts);
            
                    if fval < best
                        best = fval;
                        bestP = p;
                    end
                catch
                end
            end
            
            obj.params = bestP;
            obj.likelihood(bestP);
            
            fprintf('Best NLML: %.6f\n', best);
            end

        function [mean,var,std] = inference(obj,x)

            obj.likelihood(obj.params);

            
            [m_low, cov_low] = obj.model_low.inference(x);
            
            % High-fidelity correction terms

            k_s  = obj.RBF(obj.theta_e, x, obj.Xe);
            k_ss = obj.RBF(obj.theta_e, x, x);

            mean = obj.rho * m_low + k_s * obj.alpha;
            
            v = obj.L \ k_s';

            var = obj.rho^2 * cov_low + k_ss - (v' * v);

            std = sqrt(diag(var));

        end

        function var = variance(obj,x)

            obj.likelihood(obj.params);
            
            [~, cov_low] = obj.model_low.inference(x);
            
            % High-fidelity correction terms

            k_s  = obj.RBF(obj.theta_e, x, obj.Xe);
            k_ss = obj.RBF(obj.theta_e, x, x);
            
            v = obj.L \ k_s';

            var = obj.rho^2 * cov_low + k_ss - (v' * v);

        end

        function append(obj, X_new, yL_new, yH_new)

            obj.Xc = [obj.Xc; X_new];
            obj.yc = [obj.yc; yL_new];
            obj.Nc = size(obj.Xc,1);

            if nargin == 4
                obj.Xe = [obj.Xe; X_new];
                obj.ye = [obj.ye; yH_new];
                obj.Ne = size(obj.Xe,1);
            end

        end


        function [fid] = chooseFID(obj)

            % LF surrogate on validation set
        
            [mL_val,~] = obj.model_low.inference(obj.Xval);
        
            RMSE_L = sqrt(mean((obj.rho*mL_val - obj.ye_val).^2)) / abs(mean(obj.ye_val));
        
            delta_true = obj.ye_val - obj.rho*obj.yc_val;
        
            % discrepancy prediction

            obj.likelihood(obj.params);
        
            k_s  = obj.RBF(obj.theta_e,obj.Xval,obj.Xe);
        
            % discrepancy mean
            delta_pred = k_s * obj.alpha;

            RMSE_D = sqrt(mean((delta_pred - delta_true).^2)) / abs(mean(delta_true));
        
            fprintf('RMSE_L    = %.4e\n',RMSE_L);
            fprintf('RMSE_D = %.4e\n',RMSE_D);
        
        
            % if RMSE_L < 2*RMSE_delta && ...
            %    RMSE_L > RMSE_delta
        
            % add only low-fidelity point
            % 
            if RMSE_L > RMSE_D && RMSE_L < 2*RMSE_D
                fid = 1;
            else
                fid = 2;
            end
        end

    end
end