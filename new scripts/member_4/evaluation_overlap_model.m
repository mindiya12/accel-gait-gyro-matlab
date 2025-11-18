% ============================================================
% OVERLAP COMPARISON WITH PER-USER AUTHENTICATION METRICS
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% ============================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('OVERLAP COMPARISON: PER-USER AUTHENTICATION METRICS\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Load all three CLEANED datasets ---
configs = {
    'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

% Storage for results
results = struct();

% --- Per-User Authentication Evaluation Function ---
function [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, accuracy] = evaluatePerUser(X, y, configName)
    fprintf('--- Evaluating: %s ---\n', configName);
    fprintf('Dataset: %d samples, %d users\n', size(X, 1), length(unique(y)));
    
    % Normalize features (z-score)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;
    X_norm = (X - mu) ./ sigma;
    
    % 80/20 train-test split
    rng(42);
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
    net = train(net, X_train_t, y_train_onehot);
    
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
            % No test samples for this user (skip)
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
    
    % Remove NaN values (users with no test samples)
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

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate with per-user authentication metrics
    [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, acc] = ...
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
end

% --- Create comparison table ---
fprintf('===========================================================\n');
fprintf('PER-USER AUTHENTICATION METRICS COMPARISON\n');
fprintf('===========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
Mean_EER = [results.mean_eer]';
Std_EER = [results.std_eer]';
Mean_FAR = [results.mean_far]';
Mean_FRR = [results.mean_frr]';
Accuracy = [results.accuracy]';

comparisonTable = table(Method, NumSamples, Mean_EER, Std_EER, Mean_FAR, Mean_FRR, Accuracy);
disp(comparisonTable);

% --- Save results ---
figuresFolder = fullfile(resultsFolder, 'figures', 'overlap_comparison');
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

save(fullfile(resultsFolder, 'overlap_comparison_peruser.mat'), 'results', 'comparisonTable');

% --- Generate comparison plot ---
figure('Visible', 'off', 'Position', [100, 100, 900, 500]);

overlap_types = categorical({results.name});
overlap_types = reordercats(overlap_types, {results.name});

bar_data = [results.mean_eer]';
bar_errors = [results.std_eer]';

b = bar(overlap_types, bar_data, 'FaceColor', [0.3 0.6 0.9]);
hold on;
errorbar(overlap_types, bar_data, bar_errors, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'CapSize', 10);

ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Overlap Configuration', 'FontSize', 12, 'FontWeight', 'bold');
title('Effect of Window Overlap on Authentication Security (Per-User EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

% Add value labels
for i = 1:length(bar_data)
    text(i, bar_data(i) + bar_errors(i) + 0.3, ...
        sprintf('%.2f%%', bar_data(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, fullfile(figuresFolder, 'Overlap_Comparison_EER.png'));
close;

% --- Generate summary CSV ---
writetable(comparisonTable, fullfile(figuresFolder, 'Overlap_Comparison_Summary.csv'));

% --- Final summary ---
fprintf('\n===========================================================\n');
fprintf('OVERLAP COMPARISON COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('Best Configuration: %s\n', results(1).name);
fprintf('  Mean EER: %.2f%% ± %.2f%%\n', results(1).mean_eer, results(1).std_eer);
fprintf('  Accuracy: %.2f%%\n', results(1).accuracy);
fprintf('\nResults saved to: %s\n', figuresFolder);
fprintf('  - Numerical results: overlap_comparison_peruser.mat\n');
fprintf('  - EER comparison plot: Overlap_Comparison_EER.png\n');
fprintf('  - Summary table: Overlap_Comparison_Summary.csv\n');
fprintf('===========================================================\n\n');
