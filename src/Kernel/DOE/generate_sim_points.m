% nDOE = 5;
% 
% DOE = MakeDOE(3, nDOE, 0, 1);
% 
% lb = [5    50    0.05];
% ub = [50   100   0.5];
% 
% DOE = lb + DOE .* (ub - lb);
% 
% 
% DOE
%%
clear all;

rng(1)

Nc = 30;
Ne = 10;

Xc = MakeDOE(3, Nc, 0, 1);
Xe = MakeDOE(3, Ne, 0, 1);


available = true(Nc,1);   % LF points still free
Xc_new = Xc;

assign = zeros(Ne,1);     % store LF index assigned to each HF

figure;

scatter3(Xc(:,1), Xc(:,2), Xc(:,3), 30, 'b'); hold on;
scatter3(Xe(:,1), Xe(:,2), Xe(:,3), 60, 'r', 'filled');
title('Before assignment');
legend('LF','HF');

for i = 1:Ne

    lf_idx = find(available);
    D = pdist2(Xe(i,:), Xc(lf_idx,:));

    [~, minpos] = min(D);
    chosen = lf_idx(minpos);

    assign(i) = chosen;

    % overwrite LF point with HF point (your idea)
    Xc_new(chosen,:) = Xe(i,:);

    % remove from pool
    available(chosen) = false;

end


figure;

subplot(1,2,1)
scatter3(Xc(:,1), Xc(:,2), Xc(:,3), 30, 'b'); hold on;
scatter3(Xe(:,1), Xe(:,2), Xe(:,3), 60, 'r', 'filled');
title('Before assignment');
legend('LF','HF');

subplot(1,2,2)
scatter3(Xc_new(:,1), Xc_new(:,2), Xc_new(:,3), 30, 'b'); hold on;
scatter3(Xe(:,1), Xe(:,2), Xe(:,3), 60, 'r', 'filled');
title('After assignment (HF embedded in LF)');
legend('LF modified','HF');


lb = [5    50    0.05];
ub = [50   100   0.5];

Xc = lb + Xc .* (ub - lb);

Xe = lb + Xe .* (ub - lb);

Xc_new = lb + Xc_new .* (ub - lb);



templateFiles = ["Z:/template-input-files/template-CT-CS.inp","Z:/template-input-files/template-SD-CS.inp"];
outputFolder = "Z:/CS-inp-files";

create_input_files(templateFiles, outputFolder, Xc_new)

