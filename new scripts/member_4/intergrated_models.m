% =========================================================================
% INTEGRATION PHASE: COMBINED OPTIMIZATIONS
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% File: integration_final_PERUSER.m
% =========================================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('INTEGRATION PHASE: COMBINED OPTIMIZATIONS\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
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
y_test_baseline = y_test_B;

rng(42);
net_baseline = patternnet([50, 30], 'trainscg');
net_baseline.trainParam.epochs = 500;
net_baseline.trainParam.showWindow = false;
net_baseline.trainParam.showCommandLine = false;
net_baseline.divideMode = 'none';

tic;
[net_baseline, ~] = train(net_baseline, X_train_baseline, y_train_baseline);
time_baseline = toc;

% Predictions
y_pred_baseline = net_baseline(X_test_baseline);
y_pred_class_baseline = vec2ind(y_pred_baseline);
y_test_class_baseline = y_test_baseline(:)';
y_pred_class_baseline = y_pred_class_baseline(:)';

accuracy_baseline = 100 * sum(y_pred_class_baseline == y_test_class_baseline) / length(y_test_class_baseline);

% Calculate per-user biometric metrics
[mean_eer_base, std_eer_base, mean_far_base, mean_frr_base] = ...
    calculate_peruser_metrics(y_pred_baseline, y_test_class_baseline, 10);

fprintf('   ✅ Baseline Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_baseline);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_base, std_eer_base);
fprintf('     Mean FAR: %.2f%%\n', mean_far_base);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_base);
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
net_eff.divideMode = 'none';

tic;
[net_eff, ~] = train(net_eff, X_train_eff, y_train_baseline);
time_eff = toc;

y_pred_eff = net_eff(X_test_eff);
y_pred_class_eff = vec2ind(y_pred_eff);
y_pred_class_eff = y_pred_class_eff(:)';

accuracy_eff = 100 * sum(y_pred_class_eff == y_test_class_baseline) / length(y_test_class_baseline);

[mean_eer_eff, std_eer_eff, mean_far_eff, mean_frr_eff] = ...
    calculate_peruser_metrics(y_pred_eff, y_test_class_baseline, 10);

fprintf('  ✅ Efficiency Single Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_eff, std_eer_eff);
fprintf('     Mean FAR: %.2f%%\n', mean_far_eff);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_eff);
fprintf('     Time: %.2fs (%.1fx faster)\n\n', time_eff, time_baseline/time_eff);

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
    net_temp.divideMode = 'none';
    
    [net_temp, ~] = train(net_temp, X_train_eff, y_train_baseline);
    ensemble_preds = ensemble_preds + net_temp(X_test_eff);
end
time_eff_ens = toc;

ensemble_preds = ensemble_preds / num_models;
y_pred_class_eff_ens = vec2ind(ensemble_preds);
y_pred_class_eff_ens = y_pred_class_eff_ens(:)';

accuracy_eff_ens = 100 * sum(y_pred_class_eff_ens == y_test_class_baseline) / length(y_test_class_baseline);

[mean_eer_eff_ens, std_eer_eff_ens, mean_far_eff_ens, mean_frr_eff_ens] = ...
    calculate_peruser_metrics(ensemble_preds, y_test_class_baseline, 10);

fprintf('   ✅ Efficiency Ensemble Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_eff_ens);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_eff_ens, std_eer_eff_ens);
fprintf('     Mean FAR: %.2f%%\n', mean_far_eff_ens);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_eff_ens);
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

% Train ensemble
ensemble_preds_real = zeros(10, size(X_test_real_t, 2));

tic;
for i = 1:num_models
    rng(seeds(i));
    net_temp = patternnet([50, 30], 'trainscg');
    net_temp.trainParam.epochs = 500;
    net_temp.trainParam.showWindow = false;
    net_temp.trainParam.showCommandLine = false;
    net_temp.divideMode = 'none';
    
    [net_temp, ~] = train(net_temp, X_train_real_t, y_train_real_oh);
    ensemble_preds_real = ensemble_preds_real + net_temp(X_test_real_t);
end
time_real = toc;

ensemble_preds_real = ensemble_preds_real / num_models;
y_pred_class_real = vec2ind(ensemble_preds_real);
y_test_class_real = y_test_real(:)';
y_pred_class_real = y_pred_class_real(:)';

accuracy_real = 100 * sum(y_pred_class_real == y_test_class_real) / length(y_test_class_real);

[mean_eer_real, std_eer_real, mean_far_real, mean_frr_real] = ...
    calculate_peruser_metrics(ensemble_preds_real, y_test_class_real, 10);

fprintf('  ✅ Real-World Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_real);
fprintf('     Mean EER: %.2f%% ± %.2f%%\n', mean_eer_real, std_eer_real);
fprintf('     Mean FAR: %.2f%%\n', mean_far_real);
fprintf('     Mean FRR: %.2f%%\n', mean_frr_real);
fprintf('     Time: %.2fs\n\n', time_real);

%% ========================================================================
%  COMPREHENSIVE COMPARISON TABLE
% =========================================================================
fprintf('===========================================================\n');
fprintf('COMPREHENSIVE RESULTS (PER-USER METRICS)\n');
fprintf('===========================================================\n\n');

fprintf('%-35s | %-10s | %-15s | %-8s | %-10s\n', ...
    'Configuration', 'Accuracy', 'Mean EER', 'Features', 'Time (s)');
fprintf('%-35s | %-10s | %-15s | %-8s | %-10s\n', ...
    repmat('-',1,35), repmat('-',1,10), repmat('-',1,15), repmat('-',1,8), repmat('-',1,10));
fprintf('%-35s | %9.2f%% | %7.2f%% ± %.2f%% | %8d | %9.2f\n', ...
    'Baseline (All 133)', accuracy_baseline, mean_eer_base, std_eer_base, 133, time_baseline);
fprintf('%-35s | %9.2f%% | %7.2f%% ± %.2f%% | %8d | %9.2f\n', ...
    'Efficiency: Single (Top-20)', accuracy_eff, mean_eer_eff, std_eer_eff, 20, time_eff);
fprintf('%-35s | %9.2f%% | %7.2f%% ± %.2f%% | %8d | %9.2f\n', ...
    'Efficiency: Ensemble (Top-20)', accuracy_eff_ens, mean_eer_eff_ens, std_eer_eff_ens, 20, time_eff_ens);
fprintf('%-35s | %9.2f%% | %7.2f%% ± %.2f%% | %8d | %9.2f\n\n', ...
    'Real-World: Ensemble (Top-80)', accuracy_real, mean_eer_real, std_eer_real, 80, time_real);

%% ========================================================================
%  CREATE VISUALIZATION
% =========================================================================

% Create results folder
resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\figures\integration';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% Save numerical results
integration_results = struct();
integration_results.baseline = struct('accuracy', accuracy_baseline, ...
    'mean_EER', mean_eer_base, 'std_EER', std_eer_base, ...
    'mean_FAR', mean_far_base, 'mean_FRR', mean_frr_base, 'time', time_baseline);
integration_results.efficiency_single = struct('accuracy', accuracy_eff, ...
    'mean_EER', mean_eer_eff, 'std_EER', std_eer_eff, ...
    'mean_FAR', mean_far_eff, 'mean_FRR', mean_frr_eff, 'time', time_eff);
integration_results.efficiency_ensemble = struct('accuracy', accuracy_eff_ens, ...
    'mean_EER', mean_eer_eff_ens, 'std_EER', std_eer_eff_ens, ...
    'mean_FAR', mean_far_eff_ens, 'mean_FRR', mean_frr_eff_ens, 'time', time_eff_ens);
integration_results.real_world = struct('accuracy', accuracy_real, ...
    'mean_EER', mean_eer_real, 'std_EER', std_eer_real, ...
    'mean_FAR', mean_far_real, 'mean_FRR', mean_frr_real, 'time', time_real);

save(fullfile(resultsFolder, 'integration_results_peruser.mat'), 'integration_results');

% Generate comparison plot
models = categorical({'Baseline (133)', 'Efficiency Single (20)', ...
    'Efficiency Ensemble (20)', 'Real-World Ensemble (80)'});
models = reordercats(models, {'Baseline (133)', 'Efficiency Single (20)', ...
    'Efficiency Ensemble (20)', 'Real-World Ensemble (80)'});

eer_values = [mean_eer_base; mean_eer_eff; mean_eer_eff_ens; mean_eer_real];
eer_errors = [std_eer_base; std_eer_eff; std_eer_eff_ens; std_eer_real];

figure('Visible', 'off', 'Position', [100, 100, 1000, 500]);
b = bar(models, eer_values, 'FaceColor', [0.3 0.6 0.9]);
hold on;
errorbar(models, eer_values, eer_errors, 'k', 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'CapSize', 10);
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Model Configuration', 'FontSize', 12, 'FontWeight', 'bold');
title('Integration Phase: Model Comparison (Per-User EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);
xtickangle(15);

% Add value labels
for i = 1:length(eer_values)
    text(i, eer_values(i) + eer_errors(i) + max(eer_values)*0.05, ...
        sprintf('%.2f%%', eer_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, fullfile(resultsFolder, 'Integration_EER_Comparison.png'));
close;

% Save summary CSV
summary_table = table(models', [accuracy_baseline; accuracy_eff; accuracy_eff_ens; accuracy_real], ...
    eer_values, eer_errors, ...
    [mean_far_base; mean_far_eff; mean_far_eff_ens; mean_far_real], ...
    [mean_frr_base; mean_frr_eff; mean_frr_eff_ens; mean_frr_real], ...
    [133; 20; 20; 80], ...
    [time_baseline; time_eff; time_eff_ens; time_real], ...
    'VariableNames', {'Model', 'Accuracy', 'Mean_EER', 'Std_EER', ...
    'Mean_FAR', 'Mean_FRR', 'Features', 'Time_s'});
writetable(summary_table, fullfile(resultsFolder, 'Integration_Summary.csv'));

fprintf('===========================================================\n');
fprintf('✅ INTEGRATION COMPLETE WITH PER-USER METRICS!\n');
fprintf('===========================================================\n');
fprintf('Results saved to: %s\n', resultsFolder);
fprintf('  - Numerical results: integration_results_peruser.mat\n');
fprintf('  - EER comparison plot: Integration_EER_Comparison.png\n');
fprintf('  - Summary table: Integration_Summary.csv\n');
fprintf('===========================================================\n\n');

%% ========================================================================
%  HELPER FUNCTION
% =========================================================================
function [mean_eer, std_eer, mean_far, mean_frr] = calculate_peruser_metrics(y_pred, y_test_class, num_users)
    thresholds = 0:0.01:1;
    userEERs = zeros(num_users, 1);
    userFARs = zeros(num_users, 1);
    userFRRs = zeros(num_users, 1);
    
    for u = 1:num_users
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
    
    % Remove NaN values (users with no test samples)
    validUsers = ~isnan(userEERs);
    mean_eer = mean(userEERs(validUsers));
    std_eer = std(userEERs(validUsers));
    mean_far = mean(userFARs(validUsers));
    mean_frr = mean(userFRRs(validUsers));
end
