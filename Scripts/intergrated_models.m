% =========================================================================
% INTEGRATION PHASE: WITH BIOMETRIC METRICS (FAR/FRR/EER)
% File: integration_final_WITH_METRICS.m
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('INTEGRATION PHASE: COMBINED OPTIMIZATIONS\n');
fprintf('With Biometric Metrics (FAR, FRR, EER)\n');
fprintf('===========================================================\n\n');

%% Load data
fprintf('[STEP 1] Loading preprocessed data...\n');
load('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat');

% Define feature subsets
top20_features = [1, 5, 8, 12, 15, 20, 25, 30, 35, 40, ...
                  45, 50, 55, 60, 65, 70, 75, 80, 85, 90];
top80_features = [1:10, 15:20, 25:35, 40:50, 55:65, 70:80, 85:95, 100:110, 115:120];

%% ========================================================================
%  BASELINE MODEL
% =========================================================================
fprintf('\n[STEP 2] Testing BASELINE configuration...\n');

X_train_baseline = X_train_B_norm';
X_test_baseline = X_test_B_norm';
y_train_baseline = full(ind2vec(y_train_B'));
y_test_baseline = full(ind2vec(y_test_B'));

rng(42);
net_baseline = patternnet([50, 30], 'trainscg');
net_baseline.trainParam.epochs = 500;
net_baseline.trainParam.showWindow = false;
net_baseline.trainParam.showCommandLine = false;
net_baseline.divideParam.trainRatio = 0.70;
net_baseline.divideParam.valRatio = 0.15;
net_baseline.divideParam.testRatio = 0.15;

tic;
[net_baseline, ~] = train(net_baseline, X_train_baseline, y_train_baseline);
time_baseline = toc;

% Predictions
y_pred_baseline = net_baseline(X_test_baseline);
y_pred_class_baseline = vec2ind(y_pred_baseline);
y_test_class_baseline = vec2ind(y_test_baseline);
accuracy_baseline = 100 * sum(y_pred_class_baseline == y_test_class_baseline) / length(y_test_class_baseline);

% Calculate biometric metrics
[FAR_base, FRR_base, EER_base, ~] = calculate_biometric_metrics(y_pred_baseline, y_test_class_baseline, 10);

fprintf('   Baseline Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_baseline);
fprintf('     EER: %.2f%%\n', EER_base);
fprintf('     FAR at EER: %.2f%%\n', FAR_base(find(abs(FAR_base - FRR_base) == min(abs(FAR_base - FRR_base)), 1)));
fprintf('     FRR at EER: %.2f%%\n', FRR_base(find(abs(FAR_base - FRR_base) == min(abs(FAR_base - FRR_base)), 1)));
fprintf('     Time: %.2fs\n\n', time_baseline);

%% ========================================================================
%  EFFICIENCY: Single Model with Top-20
% =========================================================================
fprintf('[STEP 3a] EFFICIENCY: Single model (Top-20 features)...\n');

X_train_eff = X_train_B_norm(:, top20_features)';
X_test_eff = X_test_B_norm(:, top20_features)';

rng(42);
net_eff = patternnet([50, 30], 'trainscg');
net_eff.trainParam.epochs = 500;
net_eff.trainParam.showWindow = false;
net_eff.trainParam.showCommandLine = false;
net_eff.divideParam.trainRatio = 0.70;
net_eff.divideParam.valRatio = 0.15;
net_eff.divideParam.testRatio = 0.15;

tic;
[net_eff, ~] = train(net_eff, X_train_eff, y_train_baseline);
time_eff = toc;

y_pred_eff = net_eff(X_test_eff);
y_pred_class_eff = vec2ind(y_pred_eff);
accuracy_eff = 100 * sum(y_pred_class_eff == y_test_class_baseline) / length(y_test_class_baseline);

[FAR_eff, FRR_eff, EER_eff, ~] = calculate_biometric_metrics(y_pred_eff, y_test_class_baseline, 10);

fprintf('  ✅ Efficiency Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff);
fprintf('     EER: %.2f%%\n', EER_eff);
fprintf('     Time: %.2fs\n', time_eff);
fprintf('     Speed: %.1fx faster\n\n', time_baseline/time_eff);

%% ========================================================================
%  EFFICIENCY: Ensemble with Top-20
% =========================================================================
fprintf('[STEP 3b] EFFICIENCY: Ensemble (Top-20 features)...\n');

num_models = 5;
seeds = [42, 123, 456, 789, 1011];
ensemble_preds = zeros(10, size(X_test_eff, 2));

tic;
for i = 1:num_models
    rng(seeds(i));
    net_temp = patternnet([50, 30], 'trainscg');
    net_temp.trainParam.epochs = 500;
    net_temp.trainParam.showWindow = false;
    net_temp.trainParam.showCommandLine = false;
    net_temp.divideParam.trainRatio = 0.70;
    net_temp.divideParam.valRatio = 0.15;
    net_temp.divideParam.testRatio = 0.15;
    
    [net_temp, ~] = train(net_temp, X_train_eff, y_train_baseline);
    ensemble_preds = ensemble_preds + net_temp(X_test_eff);
end
time_eff_ens = toc;

ensemble_preds = ensemble_preds / num_models;
y_pred_class_eff_ens = vec2ind(ensemble_preds);
accuracy_eff_ens = 100 * sum(y_pred_class_eff_ens == y_test_class_baseline) / length(y_test_class_baseline);

[FAR_eff_ens, FRR_eff_ens, EER_eff_ens, ~] = calculate_biometric_metrics(ensemble_preds, y_test_class_baseline, 10);

fprintf('   Efficiency Ensemble Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff_ens);
fprintf('     EER: %.2f%%\n', EER_eff_ens);
fprintf('     Time: %.2fs\n\n', time_eff_ens);

%% ========================================================================
%  REAL-WORLD: Ensemble with Top-80 + Augmentation
% =========================================================================
fprintf('[STEP 4] REAL-WORLD: Ensemble (Top-80 + augmentation)...\n');

% Re-split 85/15
X_all = [X_train_B_norm; X_test_B_norm];
y_all = [y_train_B; y_test_B];

rng(42);
cv_real = cvpartition(y_all, 'HoldOut', 0.15);
X_train_real = X_all(training(cv_real), top80_features);
X_test_real = X_all(test(cv_real), top80_features);
y_train_real = y_all(training(cv_real));
y_test_real = y_all(test(cv_real));

% Add noise augmentation
X_train_real_aug = X_train_real + 0.05 * randn(size(X_train_real));

X_train_real_t = X_train_real_aug';
X_test_real_t = X_test_real';
y_train_real_oh = full(ind2vec(y_train_real'));
y_test_real_oh = full(ind2vec(y_test_real'));

% Train ensemble
ensemble_preds_real = zeros(10, size(X_test_real_t, 2));

tic;
for i = 1:num_models
    rng(seeds(i));
    net_temp = patternnet([50, 30], 'trainscg');
    net_temp.trainParam.epochs = 500;
    net_temp.trainParam.showWindow = false;
    net_temp.trainParam.showCommandLine = false;
    net_temp.divideParam.trainRatio = 0.70;
    net_temp.divideParam.valRatio = 0.15;
    net_temp.divideParam.testRatio = 0.15;
    
    [net_temp, ~] = train(net_temp, X_train_real_t, y_train_real_oh);
    ensemble_preds_real = ensemble_preds_real + net_temp(X_test_real_t);
end
time_real = toc;

ensemble_preds_real = ensemble_preds_real / num_models;
y_pred_class_real = vec2ind(ensemble_preds_real);
y_test_class_real = vec2ind(y_test_real_oh);
accuracy_real = 100 * sum(y_pred_class_real == y_test_class_real) / length(y_test_class_real);

[FAR_real, FRR_real, EER_real, ~] = calculate_biometric_metrics(ensemble_preds_real, y_test_class_real, 10);

fprintf('  ✅ Real-World Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_real);
fprintf('     EER: %.2f%%\n', EER_real);
fprintf('     Time: %.2fs\n\n', time_real);

%% ========================================================================
%  COMPREHENSIVE COMPARISON TABLE
% =========================================================================
fprintf('===========================================================\n');
fprintf('COMPREHENSIVE RESULTS WITH BIOMETRIC METRICS\n');
fprintf('===========================================================\n\n');

fprintf('%-30s | %-10s | %-8s | %-8s | %-10s\n', ...
    'Configuration', 'Accuracy', 'EER', 'Features', 'Time (s)');
fprintf('%-30s | %-10s | %-8s | %-8s | %-10s\n', ...
    repmat('-',1,30), repmat('-',1,10), repmat('-',1,8), repmat('-',1,8), repmat('-',1,10));

fprintf('%-30s | %9.2f%% | %7.2f%% | %8d | %9.2f\n', ...
    'Baseline (All 133)', accuracy_baseline, EER_base, 133, time_baseline);
fprintf('%-30s | %9.2f%% | %7.2f%% | %8d | %9.2f\n', ...
    'Efficiency: Single (Top-20)', accuracy_eff, EER_eff, 20, time_eff);
fprintf('%-30s | %9.2f%% | %7.2f%% | %8d | %9.2f\n', ...
    'Efficiency: Ensemble (Top-20)', accuracy_eff_ens, EER_eff_ens, 20, time_eff_ens);
fprintf('%-30s | %9.2f%% | %7.2f%% | %8d | %9.2f\n\n', ...
    'Real-World: Ensemble (Top-80)', accuracy_real, EER_real, 80, time_real);

% Save results
integration_results = struct();
integration_results.baseline = struct('accuracy', accuracy_baseline, 'EER', EER_base, 'FAR', FAR_base, 'FRR', FRR_base);
integration_results.efficiency_single = struct('accuracy', accuracy_eff, 'EER', EER_eff, 'FAR', FAR_eff, 'FRR', FRR_eff);
integration_results.efficiency_ensemble = struct('accuracy', accuracy_eff_ens, 'EER', EER_eff_ens, 'FAR', FAR_eff_ens, 'FRR', FRR_eff_ens);
integration_results.real_world = struct('accuracy', accuracy_real, 'EER', EER_real, 'FAR', FAR_real, 'FRR', FRR_real);

save('integration_results_WITH_METRICS.mat', 'integration_results');

fprintf('===========================================================\n');
fprintf(' INTEGRATION COMPLETE WITH BIOMETRIC METRICS!\n');
fprintf('===========================================================\n');

%% ========================================================================
%  HELPER FUNCTION
% =========================================================================
function [FAR, FRR, EER, thresholds] = calculate_biometric_metrics(y_pred_probs, y_test_class, num_classes)
    all_genuine_scores = [];
    all_impostor_scores = [];
    
    for user_id = 1:num_classes
        user_samples_idx = (y_test_class == user_id);
        
        if sum(user_samples_idx) > 0
            genuine_scores = y_pred_probs(user_id, user_samples_idx);
            all_genuine_scores = [all_genuine_scores, genuine_scores];
            
            impostor_samples_idx = ~user_samples_idx;
            if sum(impostor_samples_idx) > 0
                impostor_preds = y_pred_probs(user_id, impostor_samples_idx);
                all_impostor_scores = [all_impostor_scores, impostor_preds];
            end
        end
    end
    
    thresholds = 0:0.01:1;
    FAR = zeros(size(thresholds));
    FRR = zeros(size(thresholds));
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        FAR(i) = sum(all_impostor_scores >= thresh) / length(all_impostor_scores) * 100;
        FRR(i) = sum(all_genuine_scores < thresh) / length(all_genuine_scores) * 100;
    end
    
    [~, eer_idx] = min(abs(FAR - FRR));
    EER = (FAR(eer_idx) + FRR(eer_idx)) / 2;
end
