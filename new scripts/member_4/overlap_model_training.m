% ============================================================
% COMPARE OVERLAPPING VS NON-OVERLAPPING PERFORMANCE
% Per-User Evaluation (Genuine=1, Impostor=0) - Duplicate-Free
% ============================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('OVERLAP COMPARISON (DUPLICATE-FREE, PER-USER EVALUATION)\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Load all three CLEANED datasets ---
configs = {
    'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

fprintf('Loading cleaned datasets...\n');
for i = 1:size(configs, 1)
    filePath = fullfile(resultsFolder, configs{i, 1});
    if exist(filePath, 'file')
        load(filePath);
        fprintf('  %s: %d samples x %d features\n', configs{i, 2}, ...
                size(featureMatrix, 1), size(featureMatrix, 2));
    else
        error('Cleaned dataset not found: %s', configs{i, 1});
    end
end
fprintf('\n');

% --- Per-User Evaluation Function ---
function [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, accuracy, trainTime] = evaluatePerUser(X, y, configName)
    fprintf('--- Evaluating: %s ---\n', configName);
    fprintf('Dataset: %d samples, %d users\n', size(X, 1), length(unique(y)));
    
    % Normalize features (z-score)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;
    X_norm = (X - mu) ./ sigma;
    
    % 80/20 train-test split
    rng(42); % Fixed seed for reproducibility
    cv = cvpartition(y, 'HoldOut', 0.2);
    
    X_train = X_norm(training(cv), :);
    y_train = y(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = y(test(cv));
    
    fprintf('  Train: %d samples, Test: %d samples\n', size(X_train, 1), size(X_test, 1));
    
    % Train neural network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 500;
    net.divideMode = 'none';  % Disable auto-division
    
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    
    fprintf('  Training neural network...\n');
    tic;
    net = train(net, X_train_t, y_train_onehot);
    trainTime = toc;
    fprintf('  Training completed in %.2f seconds\n', trainTime);
    
    % Get predictions
    y_pred = net(X_test_t);  % 10 x N matrix
    y_pred_class = vec2ind(y_pred);
    y_test_class = y_test(:)';
    y_pred_class = y_pred_class(:)';
    
    % Overall accuracy
    accuracy = 100 * sum(y_pred_class == y_test_class) / length(y_test_class);
    
    % --- PER-USER EER/FAR/FRR CALCULATION ---
    numUsers = max(y);
    thresholds = 0:0.01:1;
    
    userEERs = zeros(numUsers, 1);
    userFARs = zeros(numUsers, 1);
    userFRRs = zeros(numUsers, 1);
    
    for u = 1:numUsers
        % Genuine: test samples from user u (labeled as 1)
        genuine_idx = (y_test_class == u);
        
        if sum(genuine_idx) == 0
            userEERs(u) = NaN;
            userFARs(u) = NaN;
            userFRRs(u) = NaN;
            continue;
        end
        
        genuine_scores = y_pred(u, genuine_idx);
        
        % Impostor: test samples NOT from user u (labeled as 0)
        impostor_idx = ~genuine_idx;
        impostor_scores = y_pred(u, impostor_idx);
        
        % Calculate FAR and FRR across thresholds
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
        
        % Find EER (where FAR ≈ FRR)
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end
    
    % Remove NaN values
    validUsers = ~isnan(userEERs);
    userEERs = userEERs(validUsers);
    userFARs = userFARs(validUsers);
    userFRRs = userFRRs(validUsers);
    
    % Calculate mean and std
    mean_eer = mean(userEERs);
    std_eer = std(userEERs);
    mean_far = mean(userFARs);
    std_far = std(userFARs);
    mean_frr = mean(userFRRs);
    std_frr = std(userFRRs);
    
    fprintf('  Results:\n');
    fprintf('    Accuracy: %.2f%%\n', accuracy);
    fprintf('    Mean EER: %.2f%% ± %.2f%%\n', mean_eer, std_eer);
    fprintf('    Mean FAR: %.2f%% ± %.2f%%\n', mean_far, std_far);
    fprintf('    Mean FRR: %.2f%% ± %.2f%%\n\n', mean_frr, std_frr);
end

% Storage for results
results = struct();

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate with per-user metrics
    [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, acc, time] = ...
        evaluatePerUser(featureMatrix, allLabels, configName);
    
    % Store results
    results(configIdx).name = configName;
    results(configIdx).numSamples = size(featureMatrix, 1);
    results(configIdx).mean_eer = mean_eer;
    results(configIdx).std_eer = std_eer;
    results(configIdx).mean_far = mean_far;
    results(configIdx).std_far = std_far;
    results(configIdx).mean_frr = mean_frr;
    results(configIdx).std_frr = std_frr;
    results(configIdx).accuracy = acc;
    results(configIdx).trainTime = time;
end

% --- Create comparison table ---
fprintf('===========================================================\n');
fprintf('COMPARISON SUMMARY (PER-USER METRICS)\n');
fprintf('===========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
Mean_EER = [results.mean_eer]';
Std_EER = [results.std_eer]';
Mean_FAR = [results.mean_far]';
Mean_FRR = [results.mean_frr]';
Accuracy = [results.accuracy]';
TrainingTime = [results.trainTime]';

comparisonTable = table(Method, NumSamples, Mean_EER, Std_EER, Mean_FAR, Mean_FRR, Accuracy, TrainingTime);
disp(comparisonTable);

% --- Create results folder ---
figuresFolder = fullfile(resultsFolder, 'figures', 'overlap_comparison_cleaned');
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

% --- Save results ---
save(fullfile(resultsFolder, 'overlap_comparison_cleaned_peruser.mat'), 'results', 'comparisonTable');

% --- Generate comparison plot ---
figure('Visible', 'off', 'Position', [100, 100, 1400, 500]);

overlap_types = categorical({results.name});
overlap_types = reordercats(overlap_types, {results.name});

% Plot 1: EER with error bars
subplot(1, 4, 1);
bar_data = [results.mean_eer]';
bar_errors = [results.std_eer]';
b = bar(overlap_types, bar_data, 'FaceColor', [0.3 0.6 0.9]);
hold on;
errorbar(overlap_types, bar_data, bar_errors, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);
ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Equal Error Rate', 'FontSize', 12);
grid on;
for i = 1:length(bar_data)
    text(i, bar_data(i) + bar_errors(i) + 0.5, sprintf('%.2f%%', bar_data(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

% Plot 2: Accuracy
subplot(1, 4, 2);
bar(overlap_types, Accuracy, 'FaceColor', [0.2 0.7 0.5]);
ylabel('Accuracy (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Classification Accuracy', 'FontSize', 12);
ylim([min(Accuracy)-5, 100]);
grid on;
for i = 1:length(Accuracy)
    text(i, Accuracy(i)+1, sprintf('%.1f%%', Accuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Plot 3: Number of samples
subplot(1, 4, 3);
bar(overlap_types, NumSamples, 'FaceColor', [0.8 0.5 0.2]);
ylabel('Number of Samples', 'FontSize', 11, 'FontWeight', 'bold');
title('Dataset Size (Deduplicated)', 'FontSize', 12);
grid on;
for i = 1:length(NumSamples)
    text(i, NumSamples(i)+50, sprintf('%d', NumSamples(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Plot 4: Training time
subplot(1, 4, 4);
bar(overlap_types, TrainingTime, 'FaceColor', [0.6 0.4 0.7]);
ylabel('Time (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
title('Training Time', 'FontSize', 12);
grid on;

sgtitle('Overlapping vs Non-Overlapping (Per-User Evaluation, Duplicate-Free)', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(figuresFolder, 'Overlap_Comparison_Cleaned.png'));
close;

% --- Save summary CSV ---
writetable(comparisonTable, fullfile(figuresFolder, 'Overlap_Comparison_Cleaned_Summary.csv'));

fprintf('\n===========================================================\n');
fprintf('OVERLAP COMPARISON COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('Best Configuration: %s\n', results(1).name);
fprintf('  Mean EER: %.2f%% ± %.2f%%\n', results(1).mean_eer, results(1).std_eer);
fprintf('  Accuracy: %.2f%%\n', results(1).accuracy);
fprintf('\nResults saved to: %s\n', figuresFolder);
fprintf('  - Numerical results: overlap_comparison_cleaned_peruser.mat\n');
fprintf('  - Comparison plot: Overlap_Comparison_Cleaned.png\n');
fprintf('  - Summary table: Overlap_Comparison_Cleaned_Summary.csv\n');
fprintf('===========================================================\n\n');
