% =========================================================================
% EXPERIMENT 4.2: VALIDATION SET SIZE ANALYSIS
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.2: VALIDATION SET SIZE OPTIMIZATION\n');
fprintf('Testing different validation ratios\n');
fprintf('===========================================================\n');

% --- Load data ---
load('normalized_splits_FINAL.mat');
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Test different validation ratios ---
val_ratios = [0.10, 0.15, 0.20, 0.25];
results_val = struct();

fprintf('\nTesting %d validation ratios...\n\n', length(val_ratios));

for i = 1:length(val_ratios)
    val_ratio = val_ratios(i);
    train_ratio = 0.85 - val_ratio;  % Keep test ratio at 0.15
    
    fprintf('--- Testing validation ratio = %.2f ---\n', val_ratio);
    fprintf('    (Train: %.2f, Val: %.2f, Test: 0.15)\n', train_ratio, val_ratio);
    
    % Create network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % Set data division
    net.divideParam.trainRatio = train_ratio;
    net.divideParam.valRatio = val_ratio;
    net.divideParam.testRatio = 0.15;
    
    % Train
    [net, tr] = train(net, X_train, y_train_onehot);
    
    % Evaluate
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    accuracy = 100 * mean(y_pred_class == y_test_class);
    
    % Store results
    results_val(i).val_ratio = val_ratio;
    results_val(i).train_ratio = train_ratio;
    results_val(i).accuracy = accuracy;
    results_val(i).epochs_trained = tr.num_epochs;
    results_val(i).best_epoch = tr.best_epoch;
    results_val(i).train_perf = tr.best_tperf;
    results_val(i).val_perf = tr.best_vperf;
    
    fprintf('  Accuracy: %.2f%%\n', accuracy);
    fprintf('  Epochs trained: %d (stopped at %d)\n', tr.num_epochs, tr.best_epoch);
    fprintf('  Train MSE: %.6f, Val MSE: %.6f\n\n', tr.best_tperf, tr.best_vperf);
end

% --- Save results ---
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\exp4\exp4.2_validation.mat', 'results_val', 'val_ratios');

% --- Plot results ---
figure('Name', 'Validation Set Analysis');

subplot(2,1,1);
accuracies = [results_val.accuracy];
bar(val_ratios, accuracies);
xlabel('Validation Ratio');
ylabel('Test Accuracy (%)');
title('Effect of Validation Set Size on Performance');
ylim([min(accuracies)-1, max(accuracies)+1]);
grid on;

subplot(2,1,2);
train_perfs = [results_val.train_perf];
val_perfs = [results_val.val_perf];
plot(val_ratios, train_perfs, '-o', 'LineWidth', 2, 'DisplayName', 'Training MSE');
hold on;
plot(val_ratios, val_perfs, '-s', 'LineWidth', 2, 'DisplayName', 'Validation MSE');
xlabel('Validation Ratio');
ylabel('MSE');
title('Training vs Validation Performance');
legend('Location', 'best');
grid on;

saveas(gcf, 'member4_exp2_validation_plot.png');

fprintf('✅ Experiment 4.2 Complete!\n');
fprintf('Results saved to: member4_exp2_validation.mat\n\n');
