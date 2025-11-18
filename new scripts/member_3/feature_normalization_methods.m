% NORMALIZATION METHOD PERFORMANCE: PER-USER EVALUATION 
clear; clc;
fprintf('\nTesting impact of normalization methods (PER-USER EVALUATION ONLY)...\n');
load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

norm_methods = {'zscore', 'minmax', 'robust', 'none'};
num_methods = length(norm_methods);
numUsers = 10;
thresholds = 0:0.01:1;

mean_EERs = zeros(1, num_methods);
std_EERs = zeros(1, num_methods);
mean_FARs = zeros(1, num_methods);
std_FARs = zeros(1, num_methods);
mean_FRRs = zeros(1, num_methods);
std_FRRs = zeros(1, num_methods);

resultsFolder = 'D:\src\development\accel-gait-gyro-matlab\results\figures_member3\normalization_exp';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

for i = 1:num_methods
    method = norm_methods{i};
    fprintf('Method: %s\n', method);
   
    X_norm = featureMatrix;
   
    switch method
        case 'zscore'
            mu = mean(X_norm);
            sigma = std(X_norm);
            sigma(sigma==0) = 1;
            X_norm = (X_norm - mu) ./ sigma;
        case 'minmax'
            X_min = min(X_norm);
            X_max = max(X_norm);
            range = X_max - X_min;
            range(range==0) = 1;
            X_norm = (X_norm - X_min) ./ range;
        case 'robust'
            med = median(X_norm);
            iqr_val = prctile(X_norm, 75) - prctile(X_norm, 25);
            iqr_val(iqr_val==0) = 1;
            X_norm = (X_norm - med) ./ iqr_val;
        case 'none'
            % No normalization
    end
   
    rng(2000 + i); % For reproducibility
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
   
    % PER-USER EER, FAR, FRR CALC 
    % Report in log
    fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(i), std_EERs(i));
    fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', mean_FARs(i), std_FARs(i));
    fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', mean_FRRs(i), std_FRRs(i));
end

% Plot only EER (±std) vs normalization method (REPORT FIGURE)
figure('Visible', 'off', 'Position', [120,100,900,500]);
errorbar(1:num_methods, mean_EERs, std_EERs, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.15 0.6 0.2]);
xlabel('Normalization Method', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate (EER) (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Impact of Normalization Methods on Authentication Security (EER)', 'FontSize', 13, 'FontWeight', 'bold');
xticks(1:num_methods);
xticklabels(norm_methods);
grid on;
set(gca, 'FontSize', 11);
for i = 1:num_methods
    text(i, mean_EERs(i) + std_EERs(i) + 0.3, ...
        sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
saveas(gcf, fullfile(resultsFolder, 'Normalization_EER.png'));
close;

% Save summary table
summary_table = table(norm_methods', mean_EERs(:), std_EERs(:), mean_FARs(:), std_FARs(:), mean_FRRs(:), std_FRRs(:), ...
    'VariableNames', {'NormMethod', 'Mean_EER', 'Std_EER', 'Mean_FAR', 'Std_FAR', 'Mean_FRR', 'Std_FRR'});
writetable(summary_table, fullfile(resultsFolder, 'Normalization_Summary.csv'));

fprintf('NORMALIZATION EXPERIMENTS COMPLETE!\n');
fprintf('Per-User Results (EER/FAR/FRR) saved in summary table and report plot.\n');
fprintf('Results saved to: %s\n', resultsFolder);
fprintf(' - Main plot: Normalization_EER.png\n');
fprintf(' - Summary table: Normalization_Summary.csv\n');