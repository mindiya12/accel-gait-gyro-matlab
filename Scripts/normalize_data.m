% =========================================================================
%NORMALIZATION SCRIPT
% =========================================================================
load('split_data.mat');

fprintf('==========================================================\n');
fprintf('NORMALIZING SPLITS (CORRECTED)\n');
fprintf('==========================================================\n');

% =========================================================================
% OPTION A: Normalize
% =========================================================================
fprintf('\nNormalizing Option A...\n');

% Step 1: Calculate mean and std from TRAINING data only
muA = mean(X_train_A, 1);
sigmaA = std(X_train_A, 0, 1);

% Step 2: Normalize TRAINING data
X_train_A_norm = zeros(size(X_train_A));
for col = 1:size(X_train_A, 2)
    if sigmaA(col) > 1e-10  % Prevent division by zero
        X_train_A_norm(:, col) = (X_train_A(:, col) - muA(col)) / sigmaA(col);
    else
        X_train_A_norm(:, col) = X_train_A(:, col) - muA(col);
    end
end

% Step 3: Apply SAME parameters to TEST data
X_test_A_norm = zeros(size(X_test_A));
for col = 1:size(X_test_A, 2)
    if sigmaA(col) > 1e-10
        X_test_A_norm(:, col) = (X_test_A(:, col) - muA(col)) / sigmaA(col);
    else
        X_test_A_norm(:, col) = X_test_A(:, col) - muA(col);
    end
end

fprintf('  Train normalized: mean=%.4f, std=%.4f\n', mean(X_train_A_norm(:)), std(X_train_A_norm(:)));
fprintf('  Test normalized: mean=%.4f, std=%.4f\n', mean(X_test_A_norm(:)), std(X_test_A_norm(:)));

% =========================================================================
% OPTION B: Normalize
% =========================================================================
fprintf('\nNormalizing Option B...\n');

% Step 1: Calculate mean and std from TRAINING data only
muB = mean(X_train_B, 1);
sigmaB = std(X_train_B, 0, 1);

% Step 2: Normalize TRAINING data
X_train_B_norm = zeros(size(X_train_B));
for col = 1:size(X_train_B, 2)
    if sigmaB(col) > 1e-10
        X_train_B_norm(:, col) = (X_train_B(:, col) - muB(col)) / sigmaB(col);
    else
        X_train_B_norm(:, col) = X_train_B(:, col) - muB(col);
    end
end

% Step 3: Apply SAME parameters to TEST data
X_test_B_norm = zeros(size(X_test_B));
for col = 1:size(X_test_B, 2)
    if sigmaB(col) > 1e-10
        X_test_B_norm(:, col) = (X_test_B(:, col) - muB(col)) / sigmaB(col);
    else
        X_test_B_norm(:, col) = X_test_B(:, col) - muB(col);
    end
end

fprintf('  Train normalized: mean=%.4f, std=%.4f\n', mean(X_train_B_norm(:)), std(X_train_B_norm(:)));
fprintf('  Test normalized: mean=%.4f, std=%.4f\n', mean(X_test_B_norm(:)), std(X_test_B_norm(:)));

% =========================================================================
% SAVE CORRECTED DATA
% =========================================================================
fprintf('\nSaving corrected normalized splits...\n');
save('normalized_splits.mat', ...
    'X_train_A_norm', 'y_train_A', 'X_test_A_norm', 'y_test_A', 'muA', 'sigmaA', ...
    'X_train_B_norm', 'y_train_B', 'X_test_B_norm', 'y_test_B', 'muB', 'sigmaB');

fprintf(' Saved: normalized_splits.mat\n');

% =========================================================================
% FINAL VERIFICATION
% =========================================================================
fprintf('\n==========================================================\n');
fprintf('FINAL VERIFICATION\n');
fprintf('==========================================================\n');

fprintf('\nOPTION A (User-Wise Split):\n');
fprintf('  Train: %d samples, %d features\n', size(X_train_A_norm));
fprintf('  Test: %d samples, %d features\n', size(X_test_A_norm));
fprintf('  Train users: %s\n', mat2str(unique(y_train_A)'));
fprintf('  Test users: %s\n', mat2str(unique(y_test_A)'));
fprintf('  Train stats: mean=%.4f, std=%.4f\n', mean(X_train_A_norm(:)), std(X_train_A_norm(:)));
fprintf('  Test stats: mean=%.4f, std=%.4f\n', mean(X_test_A_norm(:)), std(X_test_A_norm(:)));

fprintf('\nOPTION B (Random Split):\n');
fprintf('  Train: %d samples, %d features\n', size(X_train_B_norm));
fprintf('  Test: %d samples, %d features\n', size(X_test_B_norm));
fprintf('  Train users: %s\n', mat2str(unique(y_train_B)'));
fprintf('  Test users: %s\n', mat2str(unique(y_test_B)'));
fprintf('  Train stats: mean=%.4f, std=%.4f\n', mean(X_train_B_norm(:)), std(X_train_B_norm(:)));
fprintf('  Test stats: mean=%.4f, std=%.4f\n', mean(X_test_B_norm(:)), std(X_test_B_norm(:)));

fprintf('\n==========================================================\n');
fprintf(' NORMALIZATION COMPLETE\n');
fprintf('==========================================================\n');
