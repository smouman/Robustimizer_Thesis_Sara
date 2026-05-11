%

classdef coGPR < handle
    properties
        Xc 
        Xe 
        yc 
        ye
        Nc 
        Ne 
        dim
        N

        noise_c 
        noise_e
        noise_fix_c 
        noise_fix_e

        stab = 1e-6

        params
        bound

        id_theta_c
        id_theta_e

        theta_c 
        theta_e
        rho

        K 
        L 
        alpha
    end

    methods

        function obj = coGPR(Xc,Xe,yc,ye,noise_var_c,noise_fix_c,noise_var_e,noise_fix_e)

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

            obj.likelihood(obj.params);
        end

        function hyper = hyperparams(obj)

            hyper_c = ones(obj.dim+1,1);
            obj.id_theta_c = 1:length(hyper_c);

            for i=1:length(hyper_c)
                obj.bound{end+1} = [1e-6 Inf];
            end

            if ~isempty(obj.noise_c) && ~obj.noise_fix_c
                hyper_c = [hyper_c; obj.noise_c];
                obj.bound{end+1} = [1e-6 Inf];
            end

            hyper_e = ones(obj.dim+1,1);
            for i=1:length(hyper_e)
                obj.bound{end+1} = [1e-6 Inf];
            end

            start = length(hyper_c)+1;
            obj.id_theta_e = start : start+length(hyper_e)-1;

            if ~isempty(obj.noise_e) && ~obj.noise_fix_e
                hyper_e = [hyper_e; obj.noise_e];
                obj.bound{end+1} = [1e-6 Inf];
            end

            rho = 1;
            obj.bound{end+1} = [-Inf Inf];

            hyper = [hyper_c; hyper_e; rho];
        end


        function K = RBF(obj,hyper,X1,X2)

            if nargin < 4
                X2 = X1;
            end
        
            sf = hyper(1);
            l = hyper(2:end);
        
            X1s = X1 ./ l';
            X2s = X2 ./ l';
        
            r = bsxfun(@minus,...
                permute(X1s,[1 3 2]),...
                permute(X2s,[3 1 2]));
        
            K = sf^2 * exp(-0.5 * sum(r.^2,3));
        
        end


        function [NLML, grad] = likelihood(obj,hyper)

            NLML = obj.likelihood_value(hyper);
        
            epsFD = 1e-6;
        
            grad = zeros(length(hyper),1);
        
            for i = 1:length(hyper)
        
                hp = hyper;
                hp(i) = hp(i) + epsFD;
        
                fp = obj.likelihood_value(hp);
        
                hm = hyper;
                hm(i) = hm(i) - epsFD;
        
                fm = obj.likelihood_value(hm);
        
                grad(i) = (fp - fm)/(2*epsFD);
        
            end
        
        end


        function NLML = likelihood_value(obj,hyper)
            
                d = obj.dim;
            
                sf_c  = hyper(1);
                ell_c = hyper(2:d+1);
            
                idx = d + 2;
            
                if ~isempty(obj.noise_c) && ~obj.noise_fix_c
                    sigma_nc = hyper(idx);
                    idx = idx + 1;
                else
                    sigma_nc = 1e-6;
                end
            
                sf_e  = hyper(idx);
                ell_e = hyper(idx+1:idx+d);
            
                idx = idx + d + 1;
            
                if ~isempty(obj.noise_e) && ~obj.noise_fix_e
                    sigma_ne = hyper(idx);
                else
                    sigma_ne = 1e-6;
                end
            
                rho = hyper(end);
            
                % Kernels
            
                theta_c = [sf_c; ell_c];
                theta_e = [sf_e; ell_e];

                obj.theta_c = theta_c;
                obj.theta_e = theta_e;
                obj.rho = rho;
                            
                Kc_cc = obj.RBF(theta_c,obj.Xc,obj.Xc);
                Kc_ce = obj.RBF(theta_c,obj.Xc,obj.Xe);
                Kc_ee = obj.RBF(theta_c,obj.Xe,obj.Xe);
            
                Ke_ee = obj.RBF(theta_e,obj.Xe,obj.Xe);
            
                % Covariance matrix
            
                Kcc = Kc_cc + eye(obj.Nc)*sigma_nc^2;
            
                Kce = rho*Kc_ce;
            
                Kee = rho^2*Kc_ee + ...
                       Ke_ee + ...
                       eye(obj.Ne)*(sigma_nc^2 + sigma_ne^2);
            
                K = [Kcc Kce;
                     Kce' Kee];
            
                obj.K = K;
  
                try
                    obj.L = chol(K + eye(obj.N)*obj.stab,'lower');
                catch
                    NLML = 1e10;
                    return
                end

                y = [obj.yc; obj.ye];
            
                obj.alpha = obj.L'\(obj.L\y);

                NLML = 0.5*y'*obj.alpha + ...
                       sum(log(diag(obj.L))) + ...
                       0.5*obj.N*log(2*pi);
        end


    function optimize(obj)

            objFun = @(p) obj.likelihood(p);

            lb = cellfun(@(b)b(1),obj.bound)';
            ub = cellfun(@(b)b(2),obj.bound)';
        
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
        
            % Multi-start optimization
        
            nRestart = 10;
        
            bestObj = Inf;
            bestParam = obj.params;
        
            % fprintf('\n==== coGPR optimization ====\n');
        
            for k = 1:nRestart
        
                % fprintf('\nRestart %d / %d\n',k,nRestart);
        
                % Initial guess
                if k == 1
        
                    p0 = obj.params;
        
                else
        
                    p0 = obj.params;
        
                    % Random perturbation
                    scale = 0.5;
        
                    p0 = p0 .* exp(scale*randn(size(p0)));
        
                    p0(end) = randn;
        
                    % Respect bounds
                    p0 = max(p0,lb);
                    p0 = min(p0,ub);
        
                end

                % Optimization
        
                try
        
                    [p,fval,exitflag,output] = ...
                        fmincon(objFun,p0,[],[],[],[],lb,ub,[],opts);
        
                    % fprintf('Final NLML: %.6f\n',fval);
        
                    if fval < bestObj
        
                        bestObj = fval;
                        bestParam = p;
        
                        % fprintf(' -> New best solution found\n');
        
                    end
        
                catch ME
        
                    % fprintf('Optimization failed:\n');
                    % fprintf('%s\n',ME.message);
        
                end
        
            end
   
            obj.params = bestParam;
        
            obj.likelihood(obj.params);
        
            % fprintf('\n==== Optimization completed ====\n');
            % fprintf('Best NLML: %.6f\n',bestObj);
    end

        function [mean,var,std] = inference(obj,x,return_std)

            obj.likelihood(obj.params);

            c_c = obj.rho * obj.RBF(obj.theta_c,obj.Xc,x);

            c_e = obj.rho^2 * obj.RBF(obj.theta_c,obj.Xe,x) + ...
                  obj.RBF(obj.theta_e,obj.Xe,x);

            c = [c_c; c_e];

            mean = c' * obj.alpha;

            v = obj.L'\(obj.L\c);

            var = obj.rho^2 * obj.RBF(obj.theta_c,x) + ...
                  obj.RBF(obj.theta_e,x) - ...
                  c' * v;

            std = sqrt(diag(var));

            if nargin < 4 || ~return_std
                std = [];
            end
        end
    end
end