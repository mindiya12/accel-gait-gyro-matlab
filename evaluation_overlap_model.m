% ============================================================
% CORRECTED EVALUATION WITH PROPER AUTHENTICATION METRICS
% Calculates EER, FAR, FRR for fair comparison with baseline
% ============================================================

clear; clc;

fprintf('=== AUTHENTICATION METRICS EVALUATION (WITH EER) ===\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Load all three CLEANED datasets ---
configs = {
    'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

% Storage for results
results = struct();

% --- Evaluation function with PROPER AUTHENTICATION METRICS ---
function [eer, far_at_eer, frr_at_eer, threshold_at_eer, accuracyClass] = evaluateAuthentication(X, y, configName)
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
    net = patternnet([50, 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 500;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;
    
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    
    fprintf('  Training...\n');
    net = train(net, X_train_t, y_train_onehot);
    
    % Get output probabilities (similarity scores)
    y_pred_probs = net(X_test_t)';  % N x 10 matrix of probabilities
    y_pred_class = vec2ind(net(X_test_t));
    
    % Classification accuracy (for reference)
    accuracyClass = sum(y_pred_class' == y_test) / length(y_test);
    
    % --- CALCULATE AUTHENTICATION METRICS (EER, FAR, FRR) ---
    
    % For each test sample, get the probability/score for the TRUE class
    genuineScores = [];
    impostorScores = [];
    
    numTest = length(y_test);
    numClasses = max(y);
    
    for i = 1:numTest
        trueClass = y_test(i);
        
        % Genuine score: probability assigned to the TRUE class
        genuineScores = [genuineScores; y_pred_probs(i, trueClass)]; %#ok<AGROW>
        
        % Impostor scores: probabilities assigned to all OTHER classes
        for c = 1:numClasses
            if c ~= trueClass
                impostorScores = [impostorScores; y_pred_probs(i, c)]; %#ok<AGROW>
            end
        end
    end
    
    fprintf('  Genuine scores: %d\n', length(genuineScores));
    fprintf('  Impostor scores: %d\n', length(impostorScores));
    
    % Calculate FAR and FRR at different thresholds
    thresholds = 0:0.01:1;
    FAR = zeros(length(thresholds), 1);
    FRR = zeros(length(thresholds), 1);
    
    for t = 1:length(thresholds)
        threshold = thresholds(t);
        
        % False Acceptance Rate: impostors accepted (score >= threshold)
        FAR(t) = sum(impostorScores >= threshold) / length(impostorScores);
        
        % False Rejection Rate: genuine users rejected (score < threshold)
        FRR(t) = sum(genuineScores < threshold) / length(genuineScores);
    end
    
    % Find Equal Error Rate (where FAR = FRR)
    [eer, eerIdx] = min(abs(FAR - FRR));
    eer = (FAR(eerIdx) + FRR(eerIdx)) / 2;  % Average for EER
    threshold_at_eer = thresholds(eerIdx);
    far_at_eer = FAR(eerIdx);
    frr_at_eer = FRR(eerIdx);
    
    fprintf('  Results:\n');
    fprintf('    Classification Accuracy: %.2f%%\n', accuracyClass * 100);
    fprintf('    EER: %.2f%% at threshold=%.2f\n', eer * 100, threshold_at_eer);
    fprintf('    FAR at EER: %.2f%%\n', far_at_eer * 100);
    fprintf('    FRR at EER: %.2f%%\n\n', frr_at_eer * 100);
end

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate with proper authentication metrics
    [eer, far, frr, thresh, acc] = evaluateAuthentication(featureMatrix, allLabels, configName);
    
    % Store results
    results(configIdx).name = configName;
    results(configIdx).numSamples = size(featureMatrix, 1);
    results(configIdx).eer = eer;
    results(configIdx).far_at_eer = far;
    results(configIdx).frr_at_eer = frr;
    results(configIdx).threshold = thresh;
    results(configIdx).accuracy = acc;
end

% --- Create comparison table ---
fprintf('==========================================================\n');
fprintf('AUTHENTICATION METRICS COMPARISON\n');
fprintf('==========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
EER_Percent = [results.eer]' * 100;
FAR_Percent = [results.far_at_eer]' * 100;
FRR_Percent = [results.frr_at_eer]' * 100;
Threshold = [results.threshold]';
Accuracy = [results.accuracy]' * 100;

comparisonTable = table(Method, NumSamples, EER_Percent, FAR_Percent, FRR_Percent, Threshold, Accuracy);
disp(comparisonTable);

fprintf('\n--- Comparison with Baseline ---\n');
fprintf('Baseline (Option B) EER: 0.37%%\n');
fprintf('Current Non-Overlapping EER: %.2f%%\n', EER_Percent(1));

if EER_Percent(1) < 1.0
    fprintf('✓ Performance is comparable to baseline!\n');
else
    fprintf('⚠ Performance differs from baseline - check evaluation consistency\n');
end

% Save results
save(fullfile(resultsFolder, 'overlap_comparison_with_eer.mat'), 'results', 'comparisonTable');

fprintf('\n==========================================================\n');
fprintf('Evaluation complete with proper authentication metrics!\n');
fprintf('==========================================================\n');
