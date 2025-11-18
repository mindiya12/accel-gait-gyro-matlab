% ============================================================
% OVERLAP COMPARISON - PER-USER EVALUATION (MATCHES BASELINE)
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% ============================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('OVERLAP COMPARISON - PER-USER BASELINE METHOD\n');
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

% --- Per-User Evaluation Function (MATCHES YOUR BASELINE) ---
function [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, accuracy] = evaluatePerUserBaseline(X, y, configName)
    fprintf('--- Evaluating: %s ---\n', configName);
    
    % Normalize features (z-score)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;
    X_norm = (X - mu) ./ sigma;
    
    % 80/20 train-test split (same as your baseline)
    rng(42);
    cv = cvpartition(y, 'HoldOut', 0.2);
    
    X_train = X_norm(training(cv), :);
    y_train = y(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = y(test(cv));
    
    fprintf('  Train: %d samples, Test: %d samples\n', length(y_train), length(y_test));
    
    % --- Train Neural Network (SAME AS YOUR BASELINE) ---
    hiddenLayerSize = [50 30];
    trainFcn = 'trainscg';
    
    net = patternnet(hiddenLayerSize, trainFcn);
    net.trainParam.epochs = 500;
    net.trainParam.goal = 1e-5;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % Same division as baseline
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Prepare data
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    
    fprintf('  Training neural network...\n');
    [net, ~] = train(net, X_train_t, y_train_onehot);
    
    % --- Predict on test set ---
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

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate using per-user baseline method
    [mean_eer, std_eer, mean_far, std_far, mean_frr, std_frr, acc] = ...
        evaluatePerUserBaseline(featureMatrix, allLabels, configName);
    
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
fprintf('OVERLAP COMPARISON - PER-USER BASELINE METHOD\n');
fprintf('===========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
Accuracy = [results.accuracy]';
Mean_EER = [results.mean_eer]';
Std_EER = [results.std_eer]';
Mean_FAR = [results.mean_far]';
Mean_FRR = [results.mean_frr]';

comparisonTable = table(Method, NumSamples, Accuracy, Mean_EER, Std_EER, Mean_FAR, Mean_FRR);
disp(comparisonTable);

fprintf('\n--- Comparison with Your Baseline ---\n');
fprintf('Your Baseline (Option B) Mean EER: 0.00%%\n');
fprintf('Non-Overlapping Mean EER: %.2f%% ± %.2f%%\n', Mean_EER(1), Std_EER(1));
fprintf('25%% Overlap Mean EER: %.2f%% ± %.2f%%\n', Mean_EER(2), Std_EER(2));
fprintf('50%% Overlap Mean EER: %.2f%% ± %.2f%%\n', Mean_EER(3), Std_EER(3));

% Calculate differences
fprintf('\n--- Performance Differences from Non-Overlapping ---\n');
for i = 2:length(Mean_EER)
    diff = Mean_EER(i) - Mean_EER(1);
    fprintf('%s: %.2f%% EER ', Method{i}, Mean_EER(i));
    if abs(diff) < 0.1
        fprintf('(no significant difference)\n');
    elseif diff < 0
        fprintf('(%.2f%% improvement)\n', abs(diff));
    else
        fprintf('(%.2f%% worse)\n', diff);
    end
end

% --- Create results folder ---
figuresFolder = fullfile(resultsFolder, 'figures', 'overlap_baseline_method');
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

% --- Save results ---
save(fullfile(resultsFolder, 'overlap_comparison_baseline_peruser.mat'), 'results', 'comparisonTable');
writetable(comparisonTable, fullfile(figuresFolder, 'Overlap_Baseline_Comparison.csv'));

fprintf('\n===========================================================\n');
fprintf('OVERLAP COMPARISON COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('All configurations use identical baseline architecture:\n');
fprintf('  - Neural Network: [50, 30] layers\n');
fprintf('  - Training: trainscg\n');
fprintf('  - Division: 70%% train, 15%% val, 15%% test\n');
fprintf('  - Evaluation: Per-user EER (genuine=1, impostor=0)\n\n');
fprintf('Results saved to: %s\n', figuresFolder);
fprintf('  - Numerical results: overlap_comparison_baseline_peruser.mat\n');
fprintf('  - Summary table: Overlap_Baseline_Comparison.csv\n');
fprintf('===========================================================\n\n');
