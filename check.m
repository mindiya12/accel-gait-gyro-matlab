% check_data_leakage.m
% Detect potential data leakage between train and test sets

fprintf('=== DATA LEAKAGE DETECTION ===\n\n');

% Load normalized splits
load('normalized_splits.mat', 'X_train_B_norm', 'y_train_B', 'X_test_B_norm', 'y_test_B');

%% Check 1: Identical samples in train and test
fprintf('Check 1: Looking for identical samples...\n');
num_duplicates = 0;
duplicate_indices = [];

for i = 1:size(X_test_B_norm, 1)
    test_sample = X_test_B_norm(i, :);
    
    % Check if this test sample exists in training set
    for j = 1:size(X_train_B_norm, 1)
        if isequal(test_sample, X_train_B_norm(j, :))
            num_duplicates = num_duplicates + 1;
            duplicate_indices = [duplicate_indices; i, j];
            break;
        end
    end
end

fprintf('  Found %d exact duplicate samples between train and test\n', num_duplicates);
if num_duplicates > 0
    fprintf('  WARNING: SEVERE DATA LEAKAGE DETECTED!\n\n');
else
    fprintf('  No exact duplicates found.\n\n');
end

%% Check 2: Near-identical samples (cosine similarity > 0.99)
fprintf('Check 2: Looking for near-identical samples...\n');
num_near_duplicates = 0;

for i = 1:min(50, size(X_test_B_norm, 1))  % Check first 50 samples
    test_sample = X_test_B_norm(i, :);
    
    % Calculate cosine similarity with all training samples
    similarities = (X_train_B_norm * test_sample') ./ ...
                   (vecnorm(X_train_B_norm, 2, 2) * norm(test_sample));
    
    max_sim = max(similarities);
    if max_sim > 0.99
        num_near_duplicates = num_near_duplicates + 1;
    end
end

fprintf('  Found %d near-identical samples (out of 50 checked)\n', num_near_duplicates);
if num_near_duplicates > 10
    fprintf('  WARNING: Possible data leakage!\n\n');
else
    fprintf('  Acceptable similarity levels.\n\n');
end

%% Check 3: User overlap between train and test
fprintf('Check 3: Checking user distribution...\n');
train_users = unique(y_train_B);
test_users = unique(y_test_B);
overlapping_users = intersect(train_users, test_users);

fprintf('  Train users: %s\n', mat2str(train_users'));
fprintf('  Test users: %s\n', mat2str(test_users'));
fprintf('  Overlapping users: %d out of %d\n', length(overlapping_users), 10);

if length(overlapping_users) == length(train_users)
    fprintf('  NOTE: All users appear in both train and test (expected for Option B)\n\n');
else
    fprintf('  Users are separated (expected for Option A)\n\n');
end

%% Check 4: Verify normalization was done separately
fprintf('Check 4: Checking normalization statistics...\n');

% Load original splits before normalization
if exist('train_test_splits.mat', 'file')
    load('train_test_splits.mat', 'X_train_B', 'X_test_B');
    
    % Check if train and test have similar statistics
    train_mean = mean(X_train_B_norm(:));
    train_std = std(X_train_B_norm(:));
    test_mean = mean(X_test_B_norm(:));
    test_std = std(X_test_B_norm(:));
    
    fprintf('  Train data: mean=%.4f, std=%.4f\n', train_mean, train_std);
    fprintf('  Test data:  mean=%.4f, std=%.4f\n', test_mean, test_std);
    
    if abs(train_mean - test_mean) < 0.01 && abs(train_std - test_std) < 0.01
        fprintf('  WARNING: Train and test have identical statistics!\n');
        fprintf('  This suggests normalization might have been done on combined data.\n\n');
    else
        fprintf('  Statistics differ appropriately (normalization done separately).\n\n');
    end
else
    fprintf('  Could not find train_test_splits.mat to verify.\n\n');
end

%% Check 5: Random prediction baseline
fprintf('Check 5: Random prediction baseline comparison...\n');
random_predictions = randi([1, 10], size(y_test_B));
random_accuracy = sum(random_predictions == y_test_B) / length(y_test_B);
fprintf('  Random guessing accuracy: %.2f%%\n', random_accuracy * 100);
fprintf('  Your model accuracy: 100.00%%\n');
fprintf('  Expected accuracy for real model: 60-95%%\n\n');

%% Summary
fprintf('=== SUMMARY ===\n');
if num_duplicates > 0
    fprintf('CRITICAL ISSUE: Exact duplicate samples found in train and test sets!\n');
    fprintf('ACTION: Re-check your data splitting process.\n');
elseif num_near_duplicates > 20
    fprintf('WARNING: Many near-identical samples detected.\n');
    fprintf('ACTION: Verify your data preprocessing and splitting.\n');
else
    fprintf('The issue may be:\n');
    fprintf('  1. Dataset is too simple/small\n');
    fprintf('  2. Users have very distinct gait patterns\n');
    fprintf('  3. Subtle data leakage not detected by these tests\n');
end
