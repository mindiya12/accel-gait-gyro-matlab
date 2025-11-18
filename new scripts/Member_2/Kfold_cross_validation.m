clear; clc;
load('C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/extracted_features.mat'); % Update path as needed
X = featureMatrix; Y = allLabels;
K = 5; % or 10 for 10-fold CV
N = size(X,1); 
numUsers = numel(unique(Y));

meanEER_fold = zeros(K,1);
meanFAR_fold = zeros(K,1);
meanFRR_fold = zeros(K,1);
stdEER_fold = zeros(K,1);

% --- Custom fold assignment (random, nearly equal size) ---
fold_sizes = floor(N/K)*ones(1,K); fold_sizes(1:mod(N,K)) = fold_sizes(1:mod(N,K)) + 1;
indices = zeros(N,1);
rp = randperm(N); % Shuffle
count = 1;
for k = 1:K
    fold_len = fold_sizes(k);
    indices(rp(count:count+fold_len-1)) = k;
    count = count + fold_len;
end

for fold = 1:K
    test_idx = (indices == fold); train_idx = ~test_idx;
    X_train = X(train_idx,:); Y_train = Y(train_idx);
    X_test  = X(test_idx,:);  Y_test  = Y(test_idx);

    mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma<1e-8)=1;
    X_train_norm = (X_train-mu)./sigma; X_test_norm = (X_test-mu)./sigma;
    X_train_t = X_train_norm'; X_test_t = X_test_norm';
    y_train_onehot = full(ind2vec(Y_train'));

    net = patternnet([50,30],'trainscg');
    net.trainParam.epochs = 500; net.trainParam.showWindow = false;
    [net,~] = train(net, X_train_t, y_train_onehot);

    output = net(X_test_t); % [numUsers x #test]
    y_test_class = Y_test(:)'; % make row for indexing

    % --- Per-user EER/FAR/FRR for this fold ---
    userEERs = zeros(numUsers,1);
    userFARs = zeros(numUsers,1);
    userFRRs = zeros(numUsers,1);
    thresholds = 0:0.01:1;

    for u = 1:numUsers
        genuine_idx = (y_test_class == u);
        impostor_idx = ~genuine_idx;
        genuine_scores = output(u, genuine_idx);
        impostor_scores = output(u, impostor_idx);
        FAR = zeros(size(thresholds));
        FRR = zeros(size(thresholds));
        for t_idx = 1:length(thresholds)
            t = thresholds(t_idx);
            if ~isempty(impostor_scores)
                FAR(t_idx) = sum(impostor_scores >= t) / length(impostor_scores) * 100;
            end
            if ~isempty(genuine_scores)
                FRR(t_idx) = sum(genuine_scores < t) / length(genuine_scores) * 100;
            end
        end
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end
    meanEER_fold(fold) = mean(userEERs);
    stdEER_fold(fold)  = std(userEERs);
    meanFAR_fold(fold) = mean(userFARs);
    meanFRR_fold(fold) = mean(userFRRs);

    fprintf('Fold %d: Mean EER = %.2f%% | FAR = %.2f%% | FRR = %.2f%%\n', ...
        fold, meanEER_fold(fold), meanFAR_fold(fold), meanFRR_fold(fold));
end

overall_meanEER = mean(meanEER_fold);
overall_stdEER  = std(meanEER_fold);

fprintf('\nK-Fold CV: Mean of fold mean EER = %.2f%% (Std: %.2f%%)\n', overall_meanEER, overall_stdEER);

% ==== REPORT PLOT: Per-fold mean EER ====
figfolder = 'C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/figures/kfold_exp';
if ~exist(figfolder, 'dir'), mkdir(figfolder); end

figure('Visible','off','Position',[100, 100, 650, 400]);
bar(meanEER_fold, 'FaceColor', [0.2 0.6 0.8]);
hold on;
errorbar(1:K, meanEER_fold, stdEER_fold, 'k.', 'LineWidth', 2, 'CapSize', 10);
set(gca,'XTick',1:K,'XTickLabel',cellstr(num2str((1:K)')),'FontSize',12,'FontWeight','bold');
xlabel('Fold (K)','FontSize',13,'FontWeight','bold'); ylabel('Mean EER (%)','FontWeight','bold','FontSize',13);
title(sprintf('%d-Fold CV: Mean Per-User EER',K),'FontSize',13,'FontWeight','bold'); grid on;
saveas(gcf, fullfile(figfolder,'KFoldCV_PerUserEER.png'));
close;

% ==== Save results summary ====
T = table((1:K)', meanEER_fold, stdEER_fold, meanFAR_fold, meanFRR_fold, ...
    'VariableNames', {'Fold','Mean_EER','Std_EER','Mean_FAR','Mean_FRR'});
writetable(T, fullfile(figfolder,'KFoldCV_PerUserEER.csv'));

fprintf('Plot saved: %s\n', fullfile(figfolder,'KFoldCV_PerUserEER.png'));
fprintf('Summary table: %s\n', fullfile(figfolder,'KFoldCV_PerUserEER.csv'));
