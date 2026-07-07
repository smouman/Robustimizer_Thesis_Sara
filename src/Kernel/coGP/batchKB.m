function [X_batch, count] = batchKB(model,A,q,count)

    X_batch = zeros(q,model.dim);

    model_temp = copy(model);

    for k = 1:q

        X_new = findMEPE(model_temp,A,count);

        X_batch(k,:) = X_new;

        % Kriging believer
        mu_new = model_temp.inference(X_new);

        model_temp.Xc = [model_temp.Xc; X_new];
        model_temp.Xe = [model_temp.Xe; X_new];

        model_temp.yc = [model_temp.yc; mu_new];
        model_temp.ye = [model_temp.ye; mu_new];

        model_temp.Nc = size(model_temp.Xc,1);
        model_temp.Ne = size(model_temp.Xe,1);

        model_temp.lowreg();
        model_temp.likelihood(model_temp.params);

        count = count + 1;

    end

end