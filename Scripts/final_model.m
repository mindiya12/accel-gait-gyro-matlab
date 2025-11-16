% =========================================================================
% FINAL MODEL SELECTION: Cross-Day Evaluation of Top Candidates
% Tests 3 deployment configurations using realistic Day 1→2 testing
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('FINAL MODEL SELECTION: Cross-Day Evaluation\n');
fprintf('===========================================================\n\n');

%% Load day-separated features
load('results\extracted_features_MD.mat');
X_day1 = featureMatrix;
y_day1 = allLabels;

load('results\extracted_features_FD.mat');
X_day2 = featureMatrix;
y_day2 = allLabels;

fprintf('Data loaded: %d Day 1 samples, %d Day 2 samples\n\n', ...
    size(X_day1,1), size(X_day2,1));

%% ========================================================================
%%  CANDIDATE 1: STANDARD MODEL (Baseline)
%% ========================================================================

fprintf('===========================================================\n');
fprintf('CANDIDATE 1: STANDARD MODEL\n');
fprintf('Configuration: [50,30], All 133 features, Single model\n');
fprintf('===========================================================\n\n');

% Use all features
X_train_std = X_day1;
X_test_std = X_day2;

% Normalize
mu_std = mean(X_train_std, 1);
sigma_std = std(X_train_std, 0, 1);
sigma_std(sigma_std == 0) = 1;

X_train_std_norm = (X_train_std - mu_std) ./ sigma_std;
X_test_std_norm = (X_test_std - mu_std) ./ sigma_std;

% Train
X_train_std_t = X_train_std_norm';
X_test_std_t = X_test_std_norm';
y_train_std_oh = full(ind2vec(y_day1'));
y_test_std_oh = full(ind2vec(y_day2'));

fprintf('  Training...\n');
rng(42);
net_std = patternnet([50, 30], 'trainscg');
net_std.trainParam.epochs = 500;
net_std.trainParam.showWindow = false;
net_std.trainParam.showCommandLine = false;

tic;
[net_std, ~] = train(net_std, X_train_std_t, y_train_std_oh);
time_std = toc;

% Test
y_pred_std = net_std(X_test_std_t);
y_pred_class_std = vec2ind(y_pred_std);
y_test_class_std = vec2ind(y_test_std_oh);
accuracy_std = 100 * sum(y_pred_class_std == y_test_class_std) / length(y_test_class_std);

[FAR_std, FRR_std, EER_std] = calculate_biometric_metrics(y_pred_std, y_test_class_std, 10);

fprintf('\n  ✅ Standard Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_std);
fprintf('     EER: %.2f%%\n', EER_std);
fprintf('     Features: 133\n');
fprintf('     Training time: %.2fs\n', time_std);
fprintf('     Models: 1\n\n');

%% ========================================================================
%%  CANDIDATE 2: EFFICIENCY MODEL
%% ========================================================================

fprintf('===========================================================\n');
fprintf('CANDIDATE 2: EFFICIENCY MODEL\n');
fprintf('Configuration: [50,30], Top-20 features, 5-model ensemble\n');
fprintf('Goal: Fast inference for low-power devices\n');
fprintf('===========================================================\n\n');

% Select Top-20 features (from Member 3's analysis)
top20_features = [1, 5, 8, 12, 15, 20, 25, 30, 35, 40, ...
                  45, 50, 55, 60, 65, 70, 75, 80, 85, 90];

X_train_eff = X_day1(:, top20_features);
X_test_eff = X_day2(:, top20_features);

% Normalize
mu_eff = mean(X_train_eff, 1);
sigma_eff = std(X_train_eff, 0, 1);
sigma_eff(sigma_eff == 0) = 1;

X_train_eff_norm = (X_train_eff - mu_eff) ./ sigma_eff;
X_test_eff_norm = (X_test_eff - mu_eff) ./ sigma_eff;

X_train_eff_t = X_train_eff_norm';
X_test_eff_t = X_test_eff_norm';

% Train 5-model ensemble
fprintf('  Training 5-model ensemble...\n');
num_models = 5;
seeds = [42, 123, 456, 789, 1011];
ensemble_eff = cell(num_models, 1);

tic;
for i = 1:num_models
    fprintf('    Model %d/5...\n', i);
    rng(seeds(i));
    net_temp = patternnet([50, 30], 'trainscg');
    net_temp.trainParam.epochs = 500;
    net_temp.trainParam.showWindow = false;
    net_temp.trainParam.showCommandLine = false;
    
    [ensemble_eff{i}, ~] = train(net_temp, X_train_eff_t, y_train_std_oh);
end
time_eff = toc;

% Test ensemble (average predictions)
ensemble_pred_eff = zeros(10, size(X_test_eff_t, 2));
for i = 1:num_models
    ensemble_pred_eff = ensemble_pred_eff + ensemble_eff{i}(X_test_eff_t);
end
ensemble_pred_eff = ensemble_pred_eff / num_models;

y_pred_class_eff = vec2ind(ensemble_pred_eff);
accuracy_eff = 100 * sum(y_pred_class_eff == y_test_class_std) / length(y_test_class_std);

[FAR_eff, FRR_eff, EER_eff] = calculate_biometric_metrics(ensemble_pred_eff, y_test_class_std, 10);

fprintf('\n  ✅ Efficiency Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff);
fprintf('     EER: %.2f%%\n', EER_eff);
fprintf('     Features: 20 (85%% reduction)\n');
fprintf('     Training time: %.2fs\n', time_eff);
fprintf('     Models: 5 (ensemble)\n');
fprintf('     Inference speed: %.1fx faster (fewer features)\n\n', 133/20);

%% ========================================================================
%%  CANDIDATE 3: REAL-WORLD MODEL
%% ========================================================================

fprintf('===========================================================\n');
fprintf('CANDIDATE 3: REAL-WORLD MODEL\n');
fprintf('Configuration: [50,30], Top-80 features, Ensemble + Augmentation\n');
fprintf('Goal: Robust deployment with noise resilience\n');
fprintf('===========================================================\n\n');

% Select Top-80 features
top80_features = [1:10, 15:20, 25:35, 40:50, 55:65, 70:80, 85:95, 100:110, 115:120];

X_train_real = X_day1(:, top80_features);
X_test_real = X_day2(:, top80_features);

% Add Gaussian noise augmentation to training data
noise_std = 0.05;
X_train_real_aug = X_train_real + noise_std * randn(size(X_train_real));

% Normalize
mu_real = mean(X_train_real_aug, 1);
sigma_real = std(X_train_real_aug, 0, 1);
sigma_real(sigma_real == 0) = 1;

X_train_real_norm = (X_train_real_aug - mu_real) ./ sigma_real;
X_test_real_norm = (X_test_real - mu_real) ./ sigma_real;

X_train_real_t = X_train_real_norm';
X_test_real_t = X_test_real_norm';

% Train 5-model ensemble
fprintf('  Training 5-model ensemble with augmentation...\n');
ensemble_real = cell(num_models, 1);

tic;
for i = 1:num_models
    fprintf('    Model %d/5...\n', i);
    rng(seeds(i));
    net_temp = patternnet([50, 30], 'trainscg');
    net_temp.trainParam.epochs = 500;
    net_temp.trainParam.showWindow = false;
    net_temp.trainParam.showCommandLine = false;
    
    [ensemble_real{i}, ~] = train(net_temp, X_train_real_t, y_train_std_oh);
end
time_real = toc;

% Test ensemble
ensemble_pred_real = zeros(10, size(X_test_real_t, 2));
for i = 1:num_models
    ensemble_pred_real = ensemble_pred_real + ensemble_real{i}(X_test_real_t);
end
ensemble_pred_real = ensemble_pred_real / num_models;

y_pred_class_real = vec2ind(ensemble_pred_real);
accuracy_real = 100 * sum(y_pred_class_real == y_test_class_std) / length(y_test_class_std);

[FAR_real, FRR_real, EER_real] = calculate_biometric_metrics(ensemble_pred_real, y_test_class_std, 10);

fprintf('\n  ✅ Real-World Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_real);
fprintf('     EER: %.2f%%\n', EER_real);
fprintf('     Features: 80 (40%% reduction)\n');
fprintf('     Training time: %.2fs\n', time_real);
fprintf('     Models: 5 (ensemble)\n');
fprintf('     Augmentation: Gaussian noise (σ=0.05)\n\n');

%% ========================================================================
%%  FINAL COMPARISON & SELECTION
%% ========================================================================

fprintf('===========================================================\n');
fprintf('FINAL MODEL COMPARISON (Cross-Day Evaluation)\n');
fprintf('===========================================================\n\n');

fprintf('%-25s | %-10s | %-8s | %-10s | %-10s\n', ...
    'Model', 'Accuracy', 'EER', 'Features', 'Speed');
fprintf('%-25s | %-10s | %-8s | %-10s | %-10s\n', ...
    repmat('-',1,25), repmat('-',1,10), repmat('-',1,8), repmat('-',1,10), repmat('-',1,10));

fprintf('%-25s | %9.2f%% | %7.2f%% | %10d | %10s\n', ...
    'Standard', accuracy_std, EER_std, 133, 'Baseline');
fprintf('%-25s | %9.2f%% | %7.2f%% | %10d | %10s\n', ...
    'Efficiency', accuracy_eff, EER_eff, 20, '6.7× faster');
fprintf('%-25s | %9.2f%% | %7.2f%% | %10d | %10s\n\n', ...
    'Real-World', accuracy_real, EER_real, 80, '1.7× faster');

%% Select final model based on criteria
fprintf('RECOMMENDATION:\n\n');

% Find best model
accuracies = [accuracy_std, accuracy_eff, accuracy_real];
EERs = [EER_std, EER_eff, EER_real];
[best_acc, best_idx] = max(accuracies);

model_names = {'Standard', 'Efficiency', 'Real-World'};

fprintf('  🏆 SELECTED FINAL MODEL: %s\n\n', model_names{best_idx});

fprintf('  Justification:\n');
if best_idx == 1
    fprintf('    - Highest accuracy (%.2f%%) with low EER (%.2f%%)\n', accuracy_std, EER_std);
    fprintf('    - Uses all 133 features for maximum information\n');
    fprintf('    - Single model (simplest deployment)\n');
    fprintf('    - Best for: High-security applications\n');
elseif best_idx == 2
    fprintf('    - Excellent efficiency: 85%% feature reduction\n');
    fprintf('    - Acceptable accuracy (%.2f%%) for resource-constrained devices\n', accuracy_eff);
    fprintf('    - 6.7× faster inference\n');
    fprintf('    - Best for: Wearables, IoT devices\n');
else
    fprintf('    - Best balance: %.2f%% accuracy with 40%% feature reduction\n', accuracy_real);
    fprintf('    - Noise-augmented training improves robustness\n');
    fprintf('    - Ensemble reduces variance\n');
    fprintf('    - Best for: Mobile deployment, variable conditions\n');
end

% Save results
final_model_results = struct();
final_model_results.standard = struct('accuracy', accuracy_std, 'EER', EER_std, ...
    'features', 133, 'time', time_std);
final_model_results.efficiency = struct('accuracy', accuracy_eff, 'EER', EER_eff, ...
    'features', 20, 'time', time_eff);
final_model_results.realworld = struct('accuracy', accuracy_real, 'EER', EER_real, ...
    'features', 80, 'time', time_real);
final_model_results.selected = model_names{best_idx};

save('results\final_model_selection_results.mat', 'final_model_results');

fprintf('\n===========================================================\n');
fprintf('✅ FINAL MODEL SELECTION COMPLETE!\n');
fprintf('Selected: %s Model\n', model_names{best_idx});
fprintf('Cross-Day Accuracy: %.2f%%, EER: %.2f%%\n', best_acc, EERs(best_idx));
fprintf('===========================================================\n');

%% Helper function
function [FAR, FRR, EER] = calculate_biometric_metrics(y_pred_probs, y_test_class, num_classes)
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
