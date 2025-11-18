% =========================================================================
% EXPERIMENT 4.4: ENSEMBLE LEARNING
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% =========================================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.4: ENSEMBLE METHODS\n');
fprintf('Training multiple models and combining predictions\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n');

% --- Load data ---
load('normalized_splits_FINAL.mat');
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Configuration ---
num_models = 5;
seeds = [42, 123, 456, 789, 1011];
numUsers = 10;
thresholds = 0:0.01:1;

% Storage
ensemble_models = cell(1, num_models);
single_model_EERs = zeros(num_models, 1);
single_model_FARs = zeros(num_models, 1);
single_model_FRRs = zeros(num_models, 1);
single_model_accuracies = zeros(num_models, 1);

fprintf('\nTraining ensemble of %d models...\n\n', num_models);

% --- Train individual models ---
for i = 1:num_models
    fprintf('--- Training model %d/%d (seed=%d) ---\n', i, num_models, seeds(i));
    
    % Set random seed
    rng(seeds(i));
    
    % Create network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideMode = 'none';  % Disable auto-division
    
    % Train
    [net, tr] = train(net, X_train, y_train_onehot);
    
    % Evaluate individual model with per-user metrics
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = y_test(:)';
    y_pred_class = y_pred_class(:)';
    
    % Per-user EER/FAR/FRR for this model
    userEERs = zeros(numUsers, 1);
    userFARs = zeros(numUsers, 1);
    userFRRs = zeros(numUsers, 1);
    
    for u = 1:numUsers
        genuine_idx = (y_test_class == u);
        genuine_scores = y_pred(u, genuine_idx);
        
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
    
    % Store model and metrics
    ensemble_models{i} = net;
    single_model_EERs(i) = mean(userEERs);
    single_model_FARs(i) = mean(userFARs);
    single_model_FRRs(i) = mean(userFRRs);
    single_model_accuracies(i) = 100 * sum(y_pred_class == y_test_class) / length(y_test_class);
    
    fprintf('  Mean EER: %.2f%%\n', single_model_EERs(i));
    fprintf('  Mean FAR: %.2f%%\n', single_model_FARs(i));
    fprintf('  Mean FRR: %.2f%%\n', single_model_FRRs(i));
    fprintf('  Accuracy: %.2f%%\n\n', single_model_accuracies(i));
end

% --- Combine ensemble predictions (soft voting) ---
fprintf('--- Combining ensemble predictions ---\n');

all_predictions = zeros(10, size(X_test, 2), num_models);
for i = 1:num_models
    all_predictions(:, :, i) = ensemble_models{i}(X_test);
end

% Average predictions
ensemble_pred = mean(all_predictions, 3);
ensemble_pred_class = vec2ind(ensemble_pred);
y_test_class = y_test(:)';
ensemble_pred_class = ensemble_pred_class(:)';

% Calculate ensemble per-user EER/FAR/FRR
userEERs_ensemble = zeros(numUsers, 1);
userFARs_ensemble = zeros(numUsers, 1);
userFRRs_ensemble = zeros(numUsers, 1);

for u = 1:numUsers
    genuine_idx = (y_test_class == u);
    genuine_scores = ensemble_pred(u, genuine_idx);
    
    impostor_idx = ~genuine_idx;
    impostor_scores = ensemble_pred(u, impostor_idx);
    
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
    userEERs_ensemble(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
    userFARs_ensemble(u) = FAR(eer_idx);
    userFRRs_ensemble(u) = FRR(eer_idx);
end

ensemble_EER = mean(userEERs_ensemble);
ensemble_FAR = mean(userFARs_ensemble);
ensemble_FRR = mean(userFRRs_ensemble);
ensemble_accuracy = 100 * sum(ensemble_pred_class == y_test_class) / length(y_test_class);

fprintf('\n--- INDIVIDUAL MODELS (Average) ---\n');
fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean(single_model_EERs), std(single_model_EERs));
fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', mean(single_model_FARs), std(single_model_FARs));
fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', mean(single_model_FRRs), std(single_model_FRRs));
fprintf('  Mean Accuracy: %.2f%% ± %.2f%%\n', mean(single_model_accuracies), std(single_model_accuracies));

fprintf('\n--- ENSEMBLE MODEL ---\n');
fprintf('  Ensemble EER: %.2f%%\n', ensemble_EER);
fprintf('  Ensemble FAR: %.2f%%\n', ensemble_FAR);
fprintf('  Ensemble FRR: %.2f%%\n', ensemble_FRR);
fprintf('  Ensemble Accuracy: %.2f%%\n', ensemble_accuracy);

fprintf('\n--- IMPROVEMENT ---\n');
fprintf('  EER improvement: %.2f%%\n', mean(single_model_EERs) - ensemble_EER);
fprintf('  Accuracy improvement: %.2f%%\n\n', ensemble_accuracy - mean(single_model_accuracies));

% --- Create results folder ---
resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\figures\member4_exp4';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% --- Save results ---
results_ensemble.num_models = num_models;
results_ensemble.single_model_EERs = single_model_EERs;
results_ensemble.single_model_FARs = single_model_FARs;
results_ensemble.single_model_FRRs = single_model_FRRs;
results_ensemble.single_model_accuracies = single_model_accuracies;
results_ensemble.ensemble_EER = ensemble_EER;
results_ensemble.ensemble_FAR = ensemble_FAR;
results_ensemble.ensemble_FRR = ensemble_FRR;
results_ensemble.ensemble_accuracy = ensemble_accuracy;
results_ensemble.EER_improvement = mean(single_model_EERs) - ensemble_EER;
results_ensemble.accuracy_improvement = ensemble_accuracy - mean(single_model_accuracies);

save(fullfile(resultsFolder, 'member4_exp4_ensemble_results.mat'), 'results_ensemble');

% --- Generate ONLY essential plot for report: EER comparison ---
figure('Visible', 'off', 'Position', [100, 100, 900, 500]);

% Plot individual model EERs + ensemble EER
bar_data = [single_model_EERs; ensemble_EER]';
bar(bar_data, 'FaceColor', [0.3 0.6 0.9]);
hold on;

% Add mean line for individual models
mean_line = mean(single_model_EERs);
plot([0.5, num_models+0.5], [mean_line, mean_line], '--r', 'LineWidth', 2.5);

xlabel('Model', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Ensemble Learning: Individual Models vs Combined Ensemble', 'FontSize', 13, 'FontWeight', 'bold');
xticklabels({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Ensemble'});
grid on;
legend('EER', 'Mean Individual EER', 'Location', 'best');
set(gca, 'FontSize', 11);

% Add value labels on bars
for i = 1:length(bar_data)
    text(i, bar_data(i) + 0.05, sprintf('%.2f%%', bar_data(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, fullfile(resultsFolder, 'Ensemble_EER_Comparison.png'));
close;

% --- Save summary table ---
model_names = [cellstr(num2str((1:num_models)', 'Model %d')); {'Ensemble'}];
summary_table = table(model_names, ...
    [single_model_EERs; ensemble_EER], ...
    [single_model_FARs; ensemble_FAR], ...
    [single_model_FRRs; ensemble_FRR], ...
    [single_model_accuracies; ensemble_accuracy], ...
    'VariableNames', {'Model', 'EER', 'FAR', 'FRR', 'Accuracy'});
writetable(summary_table, fullfile(resultsFolder, 'Ensemble_Summary.csv'));

fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.4 COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('Results saved to: %s\n', resultsFolder);
fprintf('  - Numerical results: member4_exp4_ensemble_results.mat\n');
fprintf('  - EER comparison plot: Ensemble_EER_Comparison.png\n');
fprintf('  - Summary table: Ensemble_Summary.csv\n');
fprintf('===========================================================\n\n');
