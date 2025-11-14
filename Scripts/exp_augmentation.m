% =========================================================================
% EXPERIMENT 4.3: DATA AUGMENTATION WITH NOISE
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.3: DATA AUGMENTATION ANALYSIS\n');
fprintf('Adding Gaussian noise to improve robustness\n');
fprintf('===========================================================\n');

% --- Load data ---
load('normalized_splits_FINAL.mat');
X_train_orig = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Test different noise levels ---
noise_levels = [0, 0.01, 0.05, 0.10];
results_noise = struct();

fprintf('\nTesting %d noise levels...\n\n', length(noise_levels));

for i = 1:length(noise_levels)
    noise_std = noise_levels(i);
    fprintf('--- Testing noise std = %.2f ---\n', noise_std);
    
    % Add Gaussian noise to training data
    if noise_std > 0
        noise = noise_std * randn(size(X_train_orig));
        X_train_noisy = X_train_orig + noise;
    else
        X_train_noisy = X_train_orig;
    end
    
    % Create network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Train on noisy data
    [net, tr] = train(net, X_train_noisy, y_train_onehot);
    
    % Evaluate on CLEAN test data (important!)
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    accuracy = 100 * mean(y_pred_class == y_test_class);
    
    % Store results
    results_noise(i).noise_std = noise_std;
    results_noise(i).accuracy = accuracy;
    results_noise(i).final_perf = tr.best_perf;
    
    fprintf('  Accuracy on clean test: %.2f%%\n', accuracy);
    fprintf('  Training MSE: %.6f\n\n', tr.best_perf);
end

% --- Save results ---
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\exp4.3_augmentation.mat', 'results_noise', 'noise_levels');

% --- Plot results ---
figure('Name', 'Data Augmentation Analysis');

accuracies = [results_noise.accuracy];
bar(noise_levels, accuracies);
xlabel('Noise Standard Deviation');
ylabel('Test Accuracy (%)');
title('Effect of Training Data Augmentation on Robustness');
ylim([min(accuracies)-1, 100]);
grid on;
text(noise_levels, accuracies, num2str(accuracies', '%.2f%%'), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

saveas(gcf, 'member4_exp3_augmentation_plot.png');

fprintf(' Experiment 4.3 Complete!\n');
fprintf('Results saved to: member4_exp3_augmentation.mat\n\n');
