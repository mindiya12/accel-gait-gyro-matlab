% FEATURE SET EXPERIMENTS: PER-USER EVALUATION
clear; clc;
fprintf('FEATURE SET PERFORMANCE: PER-USER EVALUATION ONLY\n');

load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

feature_sets = {
    1:102,            % Time-domain features only
    103:126,          % Frequency-domain features only
    127:133,          % Correlation + magnitude features only
    [1:50, 127:133],  % Selected time-domain + correlation + magnitude
    1:133             % All features - baseline
};

numSets = length(feature_sets);
numUsers = 10;
thresholds = 0:0.01:1;

mean_EERs = zeros(1, numSets);
std_EERs = zeros(1, numSets);
mean_FARs = zeros(1, numSets);
std_FARs = zeros(1, numSets);
mean_FRRs = zeros(1, numSets);
std_FRRs = zeros(1, numSets);

resultsFolder = 'D:\src\development\accel-gait-gyro-matlab\results\figures_member3\feature_sets_exp';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

for i = 1:numSets
    fprintf('\nTesting Feature Subset %d...\n', i);
    cols = feature_sets{i};
   
    X_subset = featureMatrix(:, cols);
   
    % Normalize
    mu = mean(X_subset, 1);
    sigma = std(X_subset, [], 1);
    sigma(sigma==0) = 1;
    X_norm = (X_subset - mu) ./ sigma;
   
    % 80/20 split (fix random for reproducibility)
    rng(2025 + i);  % Ensure different split for each
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
   
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    net.divideMode = 'none';  % No auto division
   
    % Prepare data
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
   
    net = train(net, X_train_t, y_train_onehot);
    y_pred = net(X_test_t);
    y_test_class = y_test(:)';
   
    % Per-user EER, FAR, FRR 
    userEERs = zeros(numUsers, 1);
    userFARs = zeros(numUsers, 1);
    userFRRs = zeros(numUsers, 1);
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
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end

    mean_EERs(i) = mean(userEERs);
    std_EERs(i) = std(userEERs);
    mean_FARs(i) = mean(userFARs);
    std_FARs(i) = std(userFARs);
    mean_FRRs(i) = mean(userFRRs);
    std_FRRs(i) = std(userFRRs);

    % Report for logs
    fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(i), std_EERs(i));
    fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', mean_FARs(i), std_FARs(i));
    fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', mean_FRRs(i), std_FRRs(i));
end

% Essential plot: EER vs. Feature Set 
figure('Visible', 'off', 'Position', [100,100,900,500]);
errorbar(1:numSets, mean_EERs, std_EERs, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.15 0.6 0.95]);
xlabel('Feature Subset', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate (EER) (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Per-User EER for Different Feature Subsets', 'FontSize', 13, 'FontWeight', 'bold');
xticks(1:numSets);
xticklabels({'Time', 'Freq', 'Corr+Mag', 'Time+Corr', 'All'});
grid on;
set(gca, 'FontSize', 11);
for i = 1:numSets
    text(i, mean_EERs(i) + std_EERs(i) + 0.3, ...
        sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment','center', 'FontSize', 10, 'FontWeight', 'bold');
end
saveas(gcf, fullfile(resultsFolder, 'FeatureSubsets_EER.png'));
close;

% Save summary table with only required metrics
summary_table = table((1:numSets)', mean_EERs(:), std_EERs(:), mean_FARs(:), std_FARs(:), mean_FRRs(:), std_FRRs(:), ...
    'VariableNames', {'FeatureSet', 'Mean_EER', 'Std_EER', 'Mean_FAR', 'Std_FAR', 'Mean_FRR', 'Std_FRR'});
writetable(summary_table, fullfile(resultsFolder, 'FeatureSubsets_Summary.csv'));

fprintf('FEATURE SET EXPERIMENTS COMPLETE!\n');
fprintf('Per-User Results (reported in summary table and EER plot only).\n');
fprintf('Results saved to: %s\n', resultsFolder);
fprintf(' - Main plot: FeatureSubsets_EER.png\n');
fprintf(' - Summary table: FeatureSubsets_Summary.csv\n');