% TOP 20 FEATURE MODEL: PER-USER EER/FAR/FRR ONLY (Coursework Standard)

clear; clc;
load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

% List actual top 20 features:
top20_features = [80 128 56 87 12 23 49 98 37 1 2 3 4 5 6 7 8 9 10 11];

fprintf('\nTraining and evaluating model using TOP 20 features only (per-user EER)...\n');
X_top20 = featureMatrix(:, top20_features);

% Normalize (z-score)
mu = mean(X_top20, 1);
sigma = std(X_top20, [], 1);
sigma(sigma == 0) = 1;
X_norm = (X_top20 - mu) ./ sigma;

rng(2025);
cv = cvpartition(allLabels, 'HoldOut', 0.2);
X_train = X_norm(training(cv), :);
y_train = allLabels(training(cv));
X_test  = X_norm(test(cv), :);
y_test  = allLabels(test(cv));

X_train_t = X_train';
y_train_onehot = full(ind2vec(y_train'));
X_test_t = X_test';

% Neural Network
hiddenLayerSize = [50 30];
net = patternnet(hiddenLayerSize);
net.trainParam.epochs = 500;
net.trainParam.showWindow = false;
net.divideMode = 'none';

fprintf('Training network...\n');
net = train(net, X_train_t, y_train_onehot);

y_pred = net(X_test_t);
y_test_class = y_test(:)';
numUsers = 10;
thresholds = 0:0.01:1;

% Compute per-user EER, FAR, FRR
userEERs = zeros(numUsers, 1);
userFARs = zeros(numUsers, 1);
userFRRs = zeros(numUsers, 1);

for u = 1:numUsers
    genuine_idx = (y_test_class == u);
    impostor_idx = ~genuine_idx;
    genuine_scores = y_pred(u, genuine_idx);
    impostor_scores = y_pred(u, impostor_idx);
   
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

meanEER = mean(userEERs);
stdEER = std(userEERs);
meanFAR = mean(userFARs);
stdFAR = std(userFARs);
meanFRR = mean(userFRRs);
stdFRR = std(userFRRs);

fprintf('  Mean EER: %.2f%% ± %.2f%%\n', meanEER, stdEER);
fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', meanFAR, stdFAR);
fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', meanFRR, stdFRR);

% Plot: per-user EERs bar plot for the report/appendix
resultsFolder = 'D:\src\development\accel-gait-gyro-matlab\results\figures_member3\top20_exp';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

figure('Visible', 'off');
bar(userEERs, 'FaceColor', [0.15 0.7 0.85]);
xlabel('User', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Per-User EER with Top 20 Most Important Features', 'FontSize', 13, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 11);
saveas(gcf, fullfile(resultsFolder,'Top20_PerUserEER.png'));
close;

% Save all results for documentation/report
save(fullfile(resultsFolder, 'model_top20_EER.mat'), ...
    'net', 'top20_features', 'userEERs', 'userFARs', 'userFRRs', ...
    'meanEER', 'stdEER', 'meanFAR', 'stdFAR', 'meanFRR', 'stdFRR', 'mu', 'sigma');

summary_table = table((1:numUsers)', userEERs, userFARs, userFRRs, ...
    'VariableNames',{'User','EER','FAR','FRR'});
writetable(summary_table, fullfile(resultsFolder, 'Top20_PerUserEER.csv'));

fprintf('TOP 20 FEATURES MODEL (PER-USER EER) COMPLETE!\n');
fprintf('Mean per-user EER: %.2f%% ± %.2f%%\n', meanEER, stdEER);
fprintf('Plot: Top20_PerUserEER.png\n');