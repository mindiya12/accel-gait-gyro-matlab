% =========================================================================
% MEMBER 2: SPLIT RATIO OPTIMIZATION (SIMPLIFIED - CORRECT VERSION)
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 2: Train/Test Split Ratio Optimization\n');
fprintf('===========================================================\n');

% Load preprocessed data
load('normalized_splits_FINAL.mat');

% Test different split ratios by re-splitting the SAME normalized data
split_ratios = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90];
results = struct();

% Combine all data
X_all = [X_train_B_norm; X_test_B_norm];
y_all = [y_train_B; y_test_B];

fprintf('Total samples: %d\n', size(X_all, 1));
fprintf('Total users: %d\n\n', length(unique(y_all)));

for idx = 1:length(split_ratios)
    ratio = split_ratios(idx);
    fprintf('--- Testing %d%% train / %d%% test split ---\n', ...
        round(ratio*100), round((1-ratio)*100));
    
    % Set seed
    rng(42);
    
    % Split data (NO RE-NORMALIZATION NEEDED - already normalized!)
    cv = cvpartition(y_all, 'HoldOut', 1-ratio);
    X_train = X_all(training(cv), :);
    X_test = X_all(test(cv), :);
    y_train = y_all(training(cv));
    y_test = y_all(test(cv));
    
    % Transpose for network
    X_train_t = X_train';
    X_test_t = X_test';
    
    % One-hot encoding
    y_train_onehot = full(ind2vec(y_train'));
    y_test_onehot = full(ind2vec(y_test'));
    
    % Create network (same as baseline)
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Train
    [net, tr] = train(net, X_train_t, y_train_onehot);
    
    % Test (CORRECT METHOD)
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    
    % Calculate accuracy
    correct = sum(y_pred_class == y_test_class);
    accuracy = 100 * correct / length(y_test_class);
    
    % Store results
    results(idx).split_ratio = ratio;
    results(idx).train_samples = size(X_train, 1);
    results(idx).test_samples = size(X_test, 1);
    results(idx).accuracy = accuracy;
    results(idx).training_mse = tr.best_perf;
    
    fprintf('  Train: %d, Test: %d samples\n', ...
        size(X_train, 1), size(X_test, 1));
    fprintf('  Accuracy: %.2f%% (%d/%d correct)\n', ...
        accuracy, correct, length(y_test_class));
    fprintf('  Training MSE: %.6f\n\n', tr.best_perf);
end

% Summary
fprintf('===========================================================\n');
fprintf('SUMMARY\n');
fprintf('===========================================================\n\n');
fprintf('%-15s %-15s %-15s\n', 'Split', 'Train', 'Accuracy (%)');
fprintf('%-15s %-15s %-15s\n', '-----', '-----', '------------');
for idx = 1:length(results)
    fprintf('%-15s %-15d %-15.2f\n', ...
        sprintf('%d/%d', round(results(idx).split_ratio*100), ...
        round((1-results(idx).split_ratio)*100)), ...
        results(idx).train_samples, ...
        results(idx).accuracy);
end

% Find best
[best_acc, best_idx] = max([results.accuracy]);
fprintf('\nBest: %d/%d split → %.2f%% accuracy\n', ...
    round(results(best_idx).split_ratio*100), ...
    round((1-results(best_idx).split_ratio)*100), ...
    best_acc);

save('member2_exp1_CORRECT.mat', 'results');
