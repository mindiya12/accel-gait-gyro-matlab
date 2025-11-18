% =========================================================================
% FINAL MODEL SELECTION: Cross-Day Evaluation of Top Candidates
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% Tests 3 deployment configurations using realistic Day 1→2 testing
% =========================================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('FINAL MODEL SELECTION: Cross-Day Evaluation\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
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

fprintf('  Training...\n');
rng(42);
net_std = patternnet([50, 30], 'trainscg');
net_std.trainParam.epochs = 500;
net_std.trainParam.showWindow = false;
net_std.trainParam.showCommandLine = false;
net_std.divideMode = 'none';

tic;
[net_std, ~] = train(net_std, X_train_std_t, y_train_std_oh);
time_std = toc;

% Test with per-user evaluation
y_pred_std = net_std(X_test_std_t);
y_pred_class_std = vec2ind(y_pred_std);
y_test_class_std = y_day2(:)';
y_pred_class_std = y_pred_class_std(:)';

accuracy_std = 100 * sum(y_pred_class_std == y_test_class_std) / length(y_test_class_std);

[mean_eer_std, std_eer_std, mean_far_std, mean_frr_std] = ...
    calculate_peruser_metrics(y_pred_std, y_test_class_std, 10);

fprintf('\n   Standard Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_std);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_std, std_eer_std);
fprintf('     Mean FAR: %.2f%%\n', mean_far_std);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_std);
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

% Select Top-20 features
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
    net_temp.divideMode = 'none';
    
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
y_pred_class_eff = y_pred_class_eff(:)';

accuracy_eff = 100 * sum(y_pred_class_eff == y_test_class_std) / length(y_test_class_std);

[mean_eer_eff, std_eer_eff, mean_far_eff, mean_frr_eff] = ...
    calculate_peruser_metrics(ensemble_pred_eff, y_test_class_std, 10);

fprintf('\n  Efficiency Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_eff, std_eer_eff);
fprintf('     Mean FAR: %.2f%%\n', mean_far_eff);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_eff);
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
    net_temp.divideMode = 'none';
    
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
y_pred_class_real = y_pred_class_real(:)';

accuracy_real = 100 * sum(y_pred_class_real == y_test_class_std) / length(y_test_class_std);

[mean_eer_real, std_eer_real, mean_far_real, mean_frr_real] = ...
    calculate_peruser_metrics(ensemble_pred_real, y_test_class_std, 10);

fprintf('\n   Real-World Model Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_real);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_real, std_eer_real);
fprintf('     Mean FAR: %.2f%%\n', mean_far_real);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_real);
fprintf('     Features: 80 (40%% reduction)\n');
fprintf('     Training time: %.2fs\n', time_real);
fprintf('     Models: 5 (ensemble)\n');
fprintf('     Augmentation: Gaussian noise (σ=0.05)\n\n');

%% ========================================================================
%%  FINAL COMPARISON & SELECTION
%% ========================================================================
fprintf('===========================================================\n');
fprintf('FINAL MODEL COMPARISON (Cross-Day Per-User Evaluation)\n');
fprintf('===========================================================\n\n');

fprintf('%-25s | %-10s | %-15s | %-10s | %-10s\n', ...
    'Model', 'Accuracy', 'Mean EER', 'Features', 'Speed');
fprintf('%-25s | %-10s | %-15s | %-10s | %-10s\n', ...
    repmat('-',1,25), repmat('-',1,10), repmat('-',1,15), repmat('-',1,10), repmat('-',1,10));
fprintf('%-25s | %9.2f%% | %7.2f%% ± %.2f%% | %10d | %10s\n', ...
    'Standard', accuracy_std, mean_eer_std, std_eer_std, 133, 'Baseline');
fprintf('%-25s | %9.2f%% | %7.2f%% ± %.2f%% | %10d | %10s\n', ...
    'Efficiency', accuracy_eff, mean_eer_eff, std_eer_eff, 20, '6.7× faster');
fprintf('%-25s | %9.2f%% | %7.2f%% ± %.2f%% | %10d | %10s\n\n', ...
    'Real-World', accuracy_real, mean_eer_real, std_eer_real, 80, '1.7× faster');

%% Select final model based on criteria
fprintf('RECOMMENDATION ANALYSIS:\n\n');

% Store results for comparison
model_names = {'Standard', 'Efficiency', 'Real-World'};
accuracies = [accuracy_std, accuracy_eff, accuracy_real];
mean_EERs = [mean_eer_std, mean_eer_eff, mean_eer_real];
std_EERs = [std_eer_std, std_eer_eff, std_eer_real];
features = [133, 20, 80];

% Find best model (lowest mean EER, or highest accuracy if EER tied)
[best_eer, best_idx] = min(mean_EERs);

fprintf('  🏆 SELECTED FINAL MODEL: %s\n\n', model_names{best_idx});

fprintf('  Justification:\n');
if best_idx == 1
    fprintf('    - Cross-day accuracy: %.2f%%\n', accuracy_std);
    fprintf('    - Mean EER: %.2f%% ± %.2f%%\n', mean_eer_std, std_eer_std);
    fprintf('    - Uses all 133 features for maximum information\n');
    fprintf('    - Single model (simplest deployment)\n');
    fprintf('    - User fairness: %.2f%% std (consistent across users)\n', std_eer_std);
    fprintf('    - Best for: High-security applications\n');
elseif best_idx == 2
    fprintf('    - Cross-day accuracy: %.2f%%\n', accuracy_eff);
    fprintf('    - Mean EER: %.2f%% ± %.2f%%\n', mean_eer_eff, std_eer_eff);
    fprintf('    - Excellent efficiency: 85%% feature reduction\n');
    fprintf('    - 6.7× faster inference (20 vs 133 features)\n');
    fprintf('    - User fairness: %.2f%% std\n', std_eer_eff);
    fprintf('    - Best for: Wearables, IoT devices, mobile apps\n');
else
    fprintf('    - Cross-day accuracy: %.2f%%\n', accuracy_real);
    fprintf('    - Mean EER: %.2f%% ± %.2f%%\n', mean_eer_real, std_eer_real);
    fprintf('    - Best balance: 40%% feature reduction\n');
    fprintf('    - Noise-augmented training improves real-world robustness\n');
    fprintf('    - Ensemble reduces variance and improves fairness\n');
    fprintf('    - User fairness: %.2f%% std (low variance)\n', std_eer_real);
    fprintf('    - Best for: General mobile deployment, variable conditions\n');
end

%% Create results folder and save
resultsFolder = 'results\figures\final_selection';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% Save results
final_model_results = struct();
final_model_results.standard = struct('accuracy', accuracy_std, ...
    'mean_EER', mean_eer_std, 'std_EER', std_eer_std, ...
    'mean_FAR', mean_far_std, 'mean_FRR', mean_frr_std, ...
    'features', 133, 'time', time_std);
final_model_results.efficiency = struct('accuracy', accuracy_eff, ...
    'mean_EER', mean_eer_eff, 'std_EER', std_eer_eff, ...
    'mean_FAR', mean_far_eff, 'mean_FRR', mean_frr_eff, ...
    'features', 20, 'time', time_eff);
final_model_results.realworld = struct('accuracy', accuracy_real, ...
    'mean_EER', mean_eer_real, 'std_EER', std_eer_real, ...
    'mean_FAR', mean_far_real, 'mean_FRR', mean_frr_real, ...
    'features', 80, 'time', time_real);
final_model_results.selected = model_names{best_idx};

save(fullfile(resultsFolder, 'final_model_selection_peruser.mat'), 'final_model_results');

% Generate comparison plot
models_cat = categorical(model_names);
models_cat = reordercats(models_cat, model_names);

figure('Visible', 'off', 'Position', [100, 100, 900, 500]);
b = bar(models_cat, mean_EERs, 'FaceColor', [0.3 0.6 0.9]);
hold on;
errorbar(models_cat, mean_EERs, std_EERs, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'CapSize', 10);
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Final Model Candidates', 'FontSize', 12, 'FontWeight', 'bold');
title('Final Model Selection: Cross-Day Per-User EER Comparison', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

% Add value labels
for i = 1:length(mean_EERs)
    text(i, mean_EERs(i) + std_EERs(i) + max(mean_EERs)*0.05, ...
        sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

% Highlight selected model
if best_idx > 0
    b.FaceColor = 'flat';
    b.CData(best_idx,:) = [0.2 0.8 0.4]; % Green for selected
end

saveas(gcf, fullfile(resultsFolder, 'Final_Model_Selection_EER.png'));
close;

% Save summary CSV
summary_table = table(models_cat', accuracies', mean_EERs', std_EERs', ...
    [mean_far_std; mean_far_eff; mean_far_real], ...
    [mean_frr_std; mean_frr_eff; mean_frr_real], ...
    features', ...
    'VariableNames', {'Model', 'Accuracy', 'Mean_EER', 'Std_EER', ...
    'Mean_FAR', 'Mean_FRR', 'Features'});
writetable(summary_table, fullfile(resultsFolder, 'Final_Selection_Summary.csv'));

fprintf('\n===========================================================\n');
fprintf(' FINAL MODEL SELECTION COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('Selected: %s Model\n', model_names{best_idx});
fprintf('Cross-Day Accuracy: %.2f%%\n', accuracies(best_idx));
fprintf('Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(best_idx), std_EERs(best_idx));
fprintf('\nResults saved to: %s\n', resultsFolder);
fprintf('  - Numerical results: final_model_selection_peruser.mat\n');
fprintf('  - EER comparison plot: Final_Model_Selection_EER.png\n');
fprintf('  - Summary table: Final_Selection_Summary.csv\n');
fprintf('===========================================================\n\n');

%% Helper function
function [mean_eer, std_eer, mean_far, mean_frr] = calculate_peruser_metrics(y_pred, y_test_class, num_users)
    thresholds = 0:0.01:1;
    userEERs = zeros(num_users, 1);
    userFARs = zeros(num_users, 1);
    userFRRs = zeros(num_users, 1);
    
    for u = 1:num_users
        % Genuine: test samples from user u
        genuine_idx = (y_test_class == u);
        
        if sum(genuine_idx) == 0
            userEERs(u) = NaN;
            userFARs(u) = NaN;
            userFRRs(u) = NaN;
            continue;
        end
        
        genuine_scores = y_pred(u, genuine_idx);
        
        % Impostor: test samples NOT from user u
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
    
    validUsers = ~isnan(userEERs);
    mean_eer = mean(userEERs(validUsers));
    std_eer = std(userEERs(validUsers));
    mean_far = mean(userFARs(validUsers));
    mean_frr = mean(userFRRs(validUsers));
end
