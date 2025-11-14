% =========================================================================
% MEMBER 4: EXPERIMENT 4.1 - REGULARIZATION (WEIGHT DECAY) ANALYSIS
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.1: WEIGHT DECAY REGULARIZATION\n');
fprintf('Testing different regularization strengths\n');
fprintf('===========================================================\n');

% --- Load baseline data ---
load('normalized_splits_FINAL.mat');

% Prepare data
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Test different regularization values ---
reg_values = [0, 0.1, 0.3, 0.5, 0.7];
results_reg = struct();

fprintf('\nTesting %d regularization values...\n\n', length(reg_values));

for i = 1:length(reg_values)
    reg = reg_values(i);
    fprintf('--- Testing regularization = %.1f ---\n', reg);
    
    % Create and configure network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % Set regularization parameter
    net.performParam.regularization = reg;
    
    % Data division
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Train the network
    [net, tr] = train(net, X_train, y_train_onehot);
    
    % Evaluate on test set
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    
    % Calculate metrics
    accuracy = 100 * mean(y_pred_class == y_test_class);
    
    % Calculate EER (simplified - use max confidence approach)
    max_conf_test = max(y_pred, [], 1);
    
    % Store results
    results_reg(i).regularization = reg;
    results_reg(i).accuracy = accuracy;
    results_reg(i).final_perf = tr.best_perf;
    results_reg(i).mean_confidence = mean(max_conf_test);
    
    fprintf('  Accuracy: %.2f%%\n', accuracy);
    fprintf('  Final MSE: %.6f\n', tr.best_perf);
    fprintf('  Mean confidence: %.4f\n\n', mean(max_conf_test));
end

% --- Save results ---
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\exp4\member4_exp1_regularization.mat', 'results_reg', 'reg_values');

% --- Plot results ---
figure('Name', 'Regularization Analysis');

subplot(2,1,1);
accuracies = [results_reg.accuracy];
plot(reg_values, accuracies, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Regularization Parameter');
ylabel('Test Accuracy (%)');
title('Effect of Regularization on Accuracy');
grid on;

subplot(2,1,2);
mse_values = [results_reg.final_perf];
plot(reg_values, mse_values, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r');
xlabel('Regularization Parameter');
ylabel('Training MSE');
title('Effect of Regularization on Training Error');
grid on;

saveas(gcf, 'member4_exp1_regularization_plot.png');

fprintf(' Experiment 4.1 Complete!\n');
fprintf('Results saved to: member4_exp1_regularization.mat\n\n');
