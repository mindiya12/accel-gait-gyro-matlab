% Load your normalized splits
load('normalized_splits.mat');

fprintf('=== VERIFICATION ===\n\n');

% Option A
fprintf('OPTION A (User-Wise Split):\n');
fprintf('  Train: %d samples, %d features\n', size(X_train_A_norm));
fprintf('  Test: %d samples, %d features\n', size(X_test_A_norm));
fprintf('  Train users: %s\n', mat2str(unique(y_train_A)'));
fprintf('  Test users: %s\n', mat2str(unique(y_test_A)'));
fprintf('  Train stats: mean=%.4f, std=%.4f\n', mean(X_train_A_norm(:)), std(X_train_A_norm(:)));
fprintf('  Test stats: mean=%.4f, std=%.4f\n\n', mean(X_test_A_norm(:)), std(X_test_A_norm(:)));

% Option B
fprintf('OPTION B (Random Split):\n');
fprintf('  Train: %d samples, %d features\n', size(X_train_B_norm));
fprintf('  Test: %d samples, %d features\n', size(X_test_B_norm));
fprintf('  Train users: %s\n', mat2str(unique(y_train_B)'));
fprintf('  Test users: %s\n', mat2str(unique(y_test_B)'));
fprintf('  Train stats: mean=%.4f, std=%.4f\n', mean(X_train_B_norm(:)), std(X_train_B_norm(:)));
fprintf('  Test stats: mean=%.4f, std=%.4f\n', mean(X_test_B_norm(:)), std(X_test_B_norm(:)));
