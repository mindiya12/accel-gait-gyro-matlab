% =========================================================================
% EXPERIMENT 4.4: ENSEMBLE LEARNING
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.4: ENSEMBLE METHODS\n');
fprintf('Training multiple models and combining predictions\n');
fprintf('===========================================================\n');

% --- Load data ---
load('normalized_splits_FINAL.mat');
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Train ensemble of models ---
num_models = 5;
seeds = [42, 123, 456, 789, 1011];
ensemble_models = cell(1, num_models);
single_accuracies = zeros(1, num_models);

fprintf('\nTraining ensemble of %d models...\n\n', num_models);

for i = 1:num_models
    fprintf('--- Training model %d/%d (seed=%d) ---\n', i, num_models, seeds(i));
    
    % Set random seed for reproducibility
    rng(seeds(i));
    
    % Create network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Train
    [net, tr] = train(net, X_train, y_train_onehot);
    
    % Evaluate individual model
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    accuracy = 100 * mean(y_pred_class == y_test_class);
    single_accuracies(i) = accuracy;
    
    % Store model
    ensemble_models{i} = net;
    
    fprintf('  Individual accuracy: %.2f%%\n\n', accuracy);
end

% --- Combine ensemble predictions ---
fprintf('--- Combining ensemble predictions ---\n');

% Get predictions from all models
all_predictions = zeros(10, size(X_test, 2), num_models);  % 10 users × samples × models
for i = 1:num_models
    all_predictions(:, :, i) = ensemble_models{i}(X_test);
end

% Average predictions (soft voting)
ensemble_pred = mean(all_predictions, 3);
ensemble_pred_class = vec2ind(ensemble_pred);
y_test_class = vec2ind(y_test_onehot);
ensemble_accuracy = 100 * mean(ensemble_pred_class == y_test_class);

fprintf('  Mean individual accuracy: %.2f%%\n', mean(single_accuracies));
fprintf('  Std of individual accuracy: %.2f%%\n', std(single_accuracies));
fprintf('  Ensemble accuracy: %.2f%%\n', ensemble_accuracy);
fprintf('  Improvement: %.2f%%\n\n', ensemble_accuracy - mean(single_accuracies));

% --- Save results ---
results_ensemble = struct();
results_ensemble.num_models = num_models;
results_ensemble.single_accuracies = single_accuracies;
results_ensemble.mean_accuracy = mean(single_accuracies);
results_ensemble.std_accuracy = std(single_accuracies);
results_ensemble.ensemble_accuracy = ensemble_accuracy;
results_ensemble.improvement = ensemble_accuracy - mean(single_accuracies);

save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\exp4\exp4.4_ensemble.mat', 'results_ensemble', 'ensemble_models');

% --- Plot results ---
figure('Name', 'Ensemble Learning Analysis');

bar_data = [single_accuracies, ensemble_accuracy];
bar(bar_data);
xlabel('Model');
ylabel('Accuracy (%)');
title('Individual Models vs Ensemble');
xticklabels({'M1', 'M2', 'M3', 'M4', 'M5', 'Ensemble'});
ylim([min(bar_data)-1, 100]);
grid on;
hold on;
plot([1, num_models+1], [mean(single_accuracies), mean(single_accuracies)], '--r', 'LineWidth', 2);
legend('Models', 'Mean Individual');

saveas(gcf, 'member4_exp4_ensemble_plot.png');

fprintf('Experiment 4.4 Complete!\n');
fprintf('Results saved to: member4_exp4_ensemble.mat\n\n');
