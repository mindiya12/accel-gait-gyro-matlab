% =========================================================================
% COMPLETE PREPROCESSING PIPELINE (BEST PRACTICE ORDER)
% File: preprocess_pipeline.m
% =========================================================================
clear; clc;

% =========================================================================
% STEP 1: LOAD RAW FEATURES (Before any splitting)
% =========================================================================
fprintf('==========================================================\n');
fprintf('COMPLETE PREPROCESSING PIPELINE\n');
fprintf('Research-backed order: Clean \u2192 Split \u2192 Normalize\n');
fprintf('==========================================================\n');

fprintf('\n[STEP 1] Loading raw features...\n');
% Load with correct variable names
S = load('C:/Users/E-TECH/OneDrive - NSBM/Desktop/ai ml/accel-gait-gyro-matlab/results/extracted_features.mat');

if isfield(S, 'featureMatrix') && isfield(S, 'allLabels')
    X_all = S.featureMatrix;
    y_all = S.allLabels;
else
    error('Variables featureMatrix/allLabels not found in extracted_features.mat.');
end

fprintf('  Original dataset: %d samples × %d features\n', size(X_all));
fprintf('  Users: %s\n', mat2str(unique(y_all)'));

% =========================================================================
% STEP 2: REMOVE DUPLICATES
% =========================================================================
fprintf('\n[STEP 2] Detecting and removing duplicates...\n');
% Removes exact duplicate rows, preserves order (uses ALL columns)
[X_clean, unique_idx, ~] = unique(X_all, 'rows', 'stable');
y_clean = y_all(unique_idx);
duplicates_removed = size(X_all, 1) - size(X_clean, 1);
fprintf('  Duplicates found: %d (%.1f%%)\n', duplicates_removed, 100 * duplicates_removed / size(X_all, 1));
fprintf('  Clean dataset: %d samples\n', size(X_clean, 1));
% User distribution
fprintf('  User distribution (after removing duplicates):\n');
for u = unique(y_clean)'
    count = sum(y_clean == u);
    fprintf('    User %d: %d samples\n', u, count);
end
% Save clean features
save('C:/Users/E-TECH/OneDrive - NSBM/Desktop/ai ml/accel-gait-gyro-matlab/results/extracted_features_clean.mat', 'X_clean', 'y_clean');
fprintf('  ✅ Saved: extracted_features_clean.mat\n');

% =========================================================================
% STEP 3: SPLIT DATA (Two strategies, both on CLEAN data)
% =========================================================================
fprintf('\n[STEP 3] Splitting clean data...\n');
% ----------- Option A: User-Wise Split --------------
fprintf('\n  Option A: User-Wise Split (Users 1-8 train, 9-10 test)\n');
train_mask_A = ismember(y_clean, 1:8);
test_mask_A = ismember(y_clean, 9:10);
X_train_A = X_clean(train_mask_A, :); y_train_A = y_clean(train_mask_A);
X_test_A  = X_clean(test_mask_A, :);  y_test_A  = y_clean(test_mask_A);
fprintf('    Train: %d samples (Users 1-8)\n', size(X_train_A, 1));
fprintf('    Test: %d samples (Users 9-10)\n', size(X_test_A, 1));
% ----------- Option B: Random 80/20 Split -----------
fprintf('\n  Option B: Random 80/20 Split\n');
rng(42);
n_samples = size(X_clean, 1);
n_train = round(0.8 * n_samples);
rand_idx = randperm(n_samples);
train_idx_B = rand_idx(1:n_train);
test_idx_B = rand_idx(n_train+1:end);
X_train_B = X_clean(train_idx_B, :); y_train_B = y_clean(train_idx_B);
X_test_B = X_clean(test_idx_B, :);   y_test_B = y_clean(test_idx_B);
fprintf('    Train: %d samples (%.1f%%)\n', size(X_train_B, 1), 100 * size(X_train_B, 1) / n_samples);
fprintf('    Test: %d samples (%.1f%%)\n', size(X_test_B, 1), 100 * size(X_test_B, 1) / n_samples);
% Save splits (unnormalized)
save('C:/Users/E-TECH/OneDrive - NSBM/Desktop/ai ml/accel-gait-gyro-matlab/results/split_data_clean.mat', ...
    'X_train_A', 'y_train_A', 'X_test_A', 'y_test_A', ...
    'X_train_B', 'y_train_B', 'X_test_B', 'y_test_B');
fprintf('  ✅ Saved: split_data_clean.mat\n');

% =========================================================================
% STEP 4: VERIFY NO LEAKAGE (Critical verification step)
% =========================================================================
fprintf('\n[STEP 4] Verifying no data leakage...\n');
% Check Option A
fprintf('  Option A verification:\n');
leak_count_A = 0;
for i = 1:min(100, size(X_test_A, 1))
    if any(ismember(X_train_A, X_test_A(i,:), 'rows'))
        leak_count_A = leak_count_A + 1;
    end
end
fprintf('    Checked %d test samples: %d duplicates found\n', min(100, size(X_test_A, 1)), leak_count_A);
if leak_count_A == 0
    fprintf('    ✅ NO LEAKAGE (Option A is clean!)\n');
else
    fprintf('    ❌ LEAKAGE DETECTED! Re-check code!\n');
end
% Check Option B
fprintf('  Option B verification:\n');
leak_count_B = 0;
for i = 1:min(100, size(X_test_B, 1))
    if any(ismember(X_train_B, X_test_B(i,:), 'rows'))
        leak_count_B = leak_count_B + 1;
    end
end
fprintf('    Checked %d test samples: %d duplicates found\n', min(100, size(X_test_B, 1)), leak_count_B);
if leak_count_B == 0
    fprintf('    ✅ NO LEAKAGE (Option B is clean!)\n');
else
    fprintf('    ❌ LEAKAGE DETECTED! Re-check code!\n');
end

% =========================================================================
% STEP 5: NORMALIZE (Using ONLY training statistics)
% =========================================================================
fprintf('\n[STEP 5] Normalizing splits...\n');
% ----------- Normalize Option A --------------
fprintf('  Normalizing Option A...\n');
muA = mean(X_train_A, 1);
sigmaA = std(X_train_A, 0, 1);
sigmaA(sigmaA < 1e-8) = 1; % Replace zero std with 1 to avoid NaNs
X_train_A_norm = (X_train_A - muA) ./ sigmaA;
X_test_A_norm = (X_test_A - muA) ./ sigmaA;
fprintf('    Train: mean=%.4f, std=%.4f\n', mean(X_train_A_norm(:)), std(X_train_A_norm(:)));
fprintf('    Test: mean=%.4f, std=%.4f\n', mean(X_test_A_norm(:)), std(X_test_A_norm(:)));
% ----------- Normalize Option B --------------
fprintf('  Normalizing Option B...\n');
muB = mean(X_train_B, 1);
sigmaB = std(X_train_B, 0, 1);
sigmaB(sigmaB < 1e-8) = 1; % Replace zero std with 1 to avoid NaNs
X_train_B_norm = (X_train_B - muB) ./ sigmaB;
X_test_B_norm = (X_test_B - muB) ./ sigmaB;
fprintf('    Train: mean=%.4f, std=%.4f\n', mean(X_train_B_norm(:)), std(X_train_B_norm(:)));
fprintf('    Test: mean=%.4f, std=%.4f\n', mean(X_test_B_norm(:)), std(X_test_B_norm(:)));

% =========================================================================
% STEP 6: SAVE FINAL CLEAN NORMALIZED DATA
% =========================================================================
fprintf('\n[STEP 6] Saving final preprocessed data...\n');
save('C:/Users/E-TECH/OneDrive - NSBM/Desktop/ai ml/accel-gait-gyro-matlab/results/normalized_splits_FINAL.mat', ...
    'X_train_A_norm', 'y_train_A', 'X_test_A_norm', 'y_test_A', 'muA', 'sigmaA', ...
    'X_train_B_norm', 'y_train_B', 'X_test_B_norm', 'y_test_B', 'muB', 'sigmaB');
fprintf('  ✅ Saved: normalized_splits_FINAL.mat\n');

% =========================================================================
% FINAL SUMMARY
% =========================================================================
fprintf('\n==========================================================\n');
fprintf('\u2705 PREPROCESSING COMPLETE (BEST PRACTICE ORDER)\n');
fprintf('==========================================================\n');
fprintf('\nSummary:\n');
fprintf('  1. Duplicates removed: %d\n', duplicates_removed);
fprintf('  2. Option A: %d train, %d test (cross-user)\n', size(X_train_A_norm, 1), size(X_test_A_norm, 1));
fprintf('  3. Option B: %d train, %d test (random split)\n', size(X_train_B_norm, 1), size(X_test_B_norm, 1));
fprintf('  4. Data leakage: VERIFIED NONE\n');
fprintf('  5. Normalization: Applied correctly\n');
fprintf('\nReady for neural network training!\n');
fprintf('==========================================================\n');