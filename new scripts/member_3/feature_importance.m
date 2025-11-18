% SEQUENTIAL FEATURE IMPORTANCE (PER-USER EER)
clear; clc;
fprintf('\nStarting sequential feature importance (per-user EER)...\n');

load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

numFeat = size(featureMatrix, 2);
numUsers = 10;
thresholds = 0:0.01:1;

% Compute base EER with all features (per-user)
X_base = featureMatrix;
mu = mean(X_base, 1);
sigma = std(X_base, [], 1);
sigma(sigma==0) = 1;
X_norm = (X_base - mu) ./ sigma;
rng(2025);
cv = cvpartition(allLabels, 'HoldOut', 0.2);
X_train = X_norm(training(cv), :);
y_train = allLabels(training(cv));
X_test = X_norm(test(cv), :);
y_test = allLabels(test(cv));
net = patternnet([50 30]);
net.trainParam.showWindow = false;
net.trainParam.epochs = 200;
net.divideMode = 'none';
X_train_t = X_train';
y_train_onehot = full(ind2vec(y_train'));
X_test_t = X_test';
net = train(net, X_train_t, y_train_onehot);
y_pred = net(X_test_t);
y_test_class = y_test(:)';
userEERs = zeros(numUsers, 1);
for u = 1:numUsers
    genuine_idx = (y_test_class == u);
    genuine_scores = y_pred(u, genuine_idx);
    impostor_idx = ~genuine_idx;
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
end
baseEER = mean(userEERs);

fprintf('Base mean EER with all features: %.4f%%\n', baseEER);

featImpact = zeros(numFeat, 1);

for f = 1:numFeat
    fprintf('Removing feature %d / %d\n', f, numFeat);
    cols = setdiff(1:numFeat, f); % Exclude current feature

    X_subset = featureMatrix(:, cols);

    % Normalize
    mu = mean(X_subset, 1);
    sigma = std(X_subset, [], 1);
    sigma(sigma==0) = 1;
    X_norm = (X_subset - mu) ./ sigma;
   
    rng(3333+f); % For reproducibility, different split each f
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    net.divideMode = 'none';
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    net = train(net, X_train_t, y_train_onehot);
    y_pred = net(X_test_t);
    y_test_class = y_test(:)';
   
    % Compute per-user EER
    userEERs = zeros(numUsers, 1);
    for u = 1:numUsers
        genuine_idx = (y_test_class == u);
        genuine_scores = y_pred(u, genuine_idx);
        impostor_idx = ~genuine_idx;
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
    end
    EER = mean(userEERs);
    featImpact(f) = EER - baseEER;  % EER increase due to this feature's removal
   
    fprintf(' Removed feature %d, EER increase: %.4f%%\n', f, featImpact(f));
end

% Identify top 20 most important features (largest EER increase)
[~, idxs] = sort(featImpact, 'descend');
top20 = idxs(1:20);

fprintf('\nTop 20 most important features (by EER impact):\n');
disp(top20');

% Plot feature importance for top 20 features (EER increase)
figure('Visible','off', 'Position', [100,100,900,400]);
bar(featImpact(top20), 'FaceColor', [0.29 0.51 0.93]);
xlabel('Feature Index (Top 20)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\Delta EER on Removal (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Top 20 Feature Importance by Impact on EER', 'FontSize', 13, 'FontWeight', 'bold');
xticklabels(cellfun(@num2str, num2cell(top20), 'UniformOutput', false));
grid on;
set(gca, 'FontSize', 11);

resultsFolder = 'D:\src\development\accel-gait-gyro-matlab\results\figures_member3\feature_importance';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end
saveas(gcf, fullfile(resultsFolder, 'SeqFeatureImportance_EER_Top20.png'));
close;

summary_table = table((1:numFeat)', featImpact, 'VariableNames', {'FeatureIdx','EERImpact'});
writetable(summary_table, fullfile(resultsFolder, 'Sequential_Feature_Importance.csv'));

fprintf('SEQ. FEATURE IMPORTANCE (PER-USER EER) COMPLETE!\n');
fprintf('- EER plot: SeqFeatureImportance_EER_Top20.png\n');
fprintf('- Full feature table: Sequential_Feature_Importance.csv\n');