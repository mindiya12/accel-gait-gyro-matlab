% ============================================================
% CORRECTED OVERLAP EVALUATION - MATCHES BASELINE METHOD
% Uses same EER calculation as your baseline Option B
% ============================================================

clear; clc;

fprintf('=== OVERLAP COMPARISON WITH BASELINE EER METHOD ===\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Load all three CLEANED datasets ---
configs = {
    'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

% Storage for results
results = struct();

% --- Evaluation function - MATCHES YOUR BASELINE ---
function [eer, threshold_eer, far_at_eer, frr_at_eer, accuracy] = evaluateBaseline(X, y, configName)
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
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Prepare data
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));
    
    fprintf('  Training neural network...\n');
    [net, tr] = train(net, X_train_t, y_train_onehot);
    
    % --- Predict on test set ---
    y_pred = net(X_test_t);  % 10 x N matrix
    y_pred_class = vec2ind(y_pred);  % Predicted class
    y_test_class = vec2ind(y_test_onehot);  % True class
    
    % Classification accuracy
    accuracy = 100 * sum(y_pred_class == y_test_class) / length(y_test_class);
    
    % --- EER CALCULATION - EXACT COPY OF YOUR BASELINE CODE ---
    
    % Maximum confidence score per sample (YOUR METHOD)
    scores = max(y_pred, [], 1)';  % N x 1
    
    thresholds = 0:0.01:1;
    FAR = zeros(size(thresholds));
    FRR = zeros(size(thresholds));
    N = numel(y_test_class);
    
    for ti = 1:length(thresholds)
        th = thresholds(ti);
        false_accepts = 0;
        false_rejects = 0;
        total_impostor = 0;
        total_genuine = 0;
        
        for i = 1:N
            pred_label = y_pred_class(i);
            true_label = y_test_class(i);
            score_i = scores(i);
            
            % If network predicts correctly (genuine attempt)
            if pred_label == true_label
                total_genuine = total_genuine + 1;
                if score_i < th
                    false_rejects = false_rejects + 1;
                end
            else % Impostor trial (wrong user claimed)
                total_impostor = total_impostor + 1;
                if score_i >= th
                    false_accepts = false_accepts + 1;
                end
            end
        end
        
        FRR(ti) = false_rejects / max(1, total_genuine);
        FAR(ti) = false_accepts / max(1, total_impostor);
    end
    
    % Find EER (where FAR = FRR)
    diff = abs(FAR - FRR);
    [~, min_idx] = min(diff);
    eer = mean([FAR(min_idx), FRR(min_idx)]);
    threshold_eer = thresholds(min_idx);
    far_at_eer = FAR(min_idx);
    frr_at_eer = FRR(min_idx);
    
    % Display results
    fprintf('  Results:\n');
    fprintf('    Classification Accuracy: %.2f%%\n', accuracy);
    fprintf('    EER: %.2f%% at threshold=%.2f\n', eer*100, threshold_eer);
    fprintf('    FAR at EER: %.2f%%\n', far_at_eer*100);
    fprintf('    FRR at EER: %.2f%%\n\n', frr_at_eer*100);
end

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate using baseline method
    [eer, thresh, far, frr, acc] = evaluateBaseline(featureMatrix, allLabels, configName);
    
    % Store results
    results(configIdx).name = configName;
    results(configIdx).numSamples = size(featureMatrix, 1);
    results(configIdx).eer = eer;
    results(configIdx).threshold = thresh;
    results(configIdx).far_at_eer = far;
    results(configIdx).frr_at_eer = frr;
    results(configIdx).accuracy = acc;
end

% --- Create comparison table ---
fprintf('==========================================================\n');
fprintf('OVERLAP COMPARISON - BASELINE METHOD\n');
fprintf('==========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
EER_Percent = [results.eer]' * 100;
FAR_Percent = [results.far_at_eer]' * 100;
FRR_Percent = [results.frr_at_eer]' * 100;
Threshold = [results.threshold]';
Accuracy = [results.accuracy]';

comparisonTable = table(Method, NumSamples, Accuracy, EER_Percent, FAR_Percent, FRR_Percent, Threshold);
disp(comparisonTable);

fprintf('\n--- Comparison with Your Baseline ---\n');
fprintf('Your Baseline EER: 0.37%%\n');
fprintf('Non-Overlapping EER: %.2f%%\n', EER_Percent(1));
fprintf('25%% Overlap EER: %.2f%%\n', EER_Percent(2));
fprintf('50%% Overlap EER: %.2f%%\n', EER_Percent(3));

% Calculate differences
fprintf('\n--- Performance Differences from Non-Overlapping ---\n');
for i = 2:length(EER_Percent)
    diff = EER_Percent(i) - EER_Percent(1);
    fprintf('%s: %.2f%% EER ', Method{i}, EER_Percent(i));
    if abs(diff) < 0.1
        fprintf('(no significant difference)\n');
    elseif diff < 0
        fprintf('(%.2f%% improvement)\n', abs(diff));
    else
        fprintf('(%.2f%% worse)\n', diff);
    end
end

% --- Save results ---
save(fullfile(resultsFolder, 'overlap_comparison_baseline_method.mat'), ...
     'results', 'comparisonTable');

fprintf('\n==========================================================\n');
fprintf('Evaluation complete using baseline EER method!\n');
fprintf('Results saved to: overlap_comparison_baseline_method.mat\n');
fprintf('==========================================================\n');
