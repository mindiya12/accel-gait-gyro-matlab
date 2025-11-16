% =========================================================================
% CROSS-DAY EVALUATION: THREE TESTING SCENARIOS
% Uses your existing 133-feature extraction
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('CROSS-DAY EVALUATION: THREE TESTING SCENARIOS\n');
fprintf('===========================================================\n\n');

%% ========================================================================
%  STEP 1: EXTRACT FEATURES SEPARATED BY DAY
% =========================================================================

fprintf('[STEP 1] Extracting features separately for Monday and Friday...\n\n');

% Set paths
data_dir = 'data\';

% Check if already extracted
if exist('results\extracted_features_MD.mat', 'file') && ...
   exist('results\extracted_features_FD.mat', 'file')
    
    fprintf('  Loading pre-extracted day-separated features...\n');
    load('results\extracted_features_MD.mat');
    X_day1 = featureMatrix;
    y_day1 = allLabels;
    
    load('results\extracted_features_FD.mat');
    X_day2 = featureMatrix;
    y_day2 = allLabels;
    
else
    % ===== EXTRACT MONDAY DATA =====
    fprintf('  Extracting Monday (MD) data...\n');
    all_features_md = [];
    all_labels_md = [];
    
    for user_id = 1:10
        filename = sprintf('U%dNW_MD.csv', user_id);
        filepath = fullfile(data_dir, filename);
        
        if ~exist(filepath, 'file')
            warning('File not found: %s', filepath);
            continue;
        end
        
        % Load raw data
        data = readmatrix(filepath);
        
        % Segment (155 samples, no overlap)
        segment_length = 155;
        num_segments = floor(size(data, 1) / segment_length);
        
        user_features = [];
        for seg_idx = 1:num_segments
            start_idx = (seg_idx - 1) * segment_length + 1;
            end_idx = seg_idx * segment_length;
            segment = data(start_idx:end_idx, :);
            
            % Extract 133 features
            features = extractFeatures_133(segment);
            user_features = [user_features; features];
        end
        
        all_features_md = [all_features_md; user_features];
        all_labels_md = [all_labels_md; user_id * ones(size(user_features, 1), 1)];
        
        fprintf('    User %d: %d segments\n', user_id, size(user_features, 1));
    end
    
    X_day1 = all_features_md;
    y_day1 = all_labels_md;
    
    % Save
    featureMatrix = X_day1;
    allLabels = y_day1;
    save('results\extracted_features_MD.mat', 'featureMatrix', 'allLabels');
    
    % ===== EXTRACT FRIDAY DATA =====
    fprintf('\n  Extracting Friday (FD) data...\n');
    all_features_fd = [];
    all_labels_fd = [];
    
    for user_id = 1:10
        filename = sprintf('U%dNW_FD.csv', user_id);
        filepath = fullfile(data_dir, filename);
        
        if ~exist(filepath, 'file')
            warning('File not found: %s', filepath);
            continue;
        end
        
        data = readmatrix(filepath);
        segment_length = 155;
        num_segments = floor(size(data, 1) / segment_length);
        
        user_features = [];
        for seg_idx = 1:num_segments
            start_idx = (seg_idx - 1) * segment_length + 1;
            end_idx = seg_idx * segment_length;
            segment = data(start_idx:end_idx, :);
            
            features = extractFeatures_133(segment);
            user_features = [user_features; features];
        end
        
        all_features_fd = [all_features_fd; user_features];
        all_labels_fd = [all_labels_fd; user_id * ones(size(user_features, 1), 1)];
        
        fprintf('    User %d: %d segments\n', user_id, size(user_features, 1));
    end
    
    X_day2 = all_features_fd;
    y_day2 = all_labels_fd;
    
    % Save
    featureMatrix = X_day2;
    allLabels = y_day2;
    save('results\extracted_features_FD.mat', 'featureMatrix', 'allLabels');
end

fprintf('\n  Day 1 (Monday): %d samples, %d features\n', size(X_day1, 1), size(X_day1, 2));
fprintf('  Day 2 (Friday): %d samples, %d features\n\n', size(X_day2, 1), size(X_day2, 2));

%% ========================================================================
%  SCENARIO 1: DAY 1 ONLY
% =========================================================================

fprintf('===========================================================\n');
fprintf('SCENARIO 1: Single Day (Day 1 train/test)\n');
fprintf('===========================================================\n\n');

% 80/20 split
rng(42);
cv1 = cvpartition(y_day1, 'HoldOut', 0.20);

X_train_s1 = X_day1(training(cv1), :);
X_test_s1 = X_day1(test(cv1), :);
y_train_s1 = y_day1(training(cv1));
y_test_s1 = y_day1(test(cv1));

% Normalize
mu_s1 = mean(X_train_s1, 1);
sigma_s1 = std(X_train_s1, 0, 1);
sigma_s1(sigma_s1 == 0) = 1;

X_train_s1_norm = (X_train_s1 - mu_s1) ./ sigma_s1;
X_test_s1_norm = (X_test_s1 - mu_s1) ./ sigma_s1;

% Transpose
X_train_s1_t = X_train_s1_norm';
X_test_s1_t = X_test_s1_norm';
y_train_s1_oh = full(ind2vec(y_train_s1'));
y_test_s1_oh = full(ind2vec(y_test_s1'));

% Train
fprintf('  Training neural network...\n');
net_s1 = patternnet([50, 30], 'trainscg');
net_s1.trainParam.epochs = 500;
net_s1.trainParam.showWindow = false;
net_s1.trainParam.showCommandLine = false;

tic;
[net_s1, ~] = train(net_s1, X_train_s1_t, y_train_s1_oh);
time_s1 = toc;

% Test
y_pred_s1 = net_s1(X_test_s1_t);
y_pred_class_s1 = vec2ind(y_pred_s1);
y_test_class_s1 = vec2ind(y_test_s1_oh);
accuracy_s1 = 100 * sum(y_pred_class_s1 == y_test_class_s1) / length(y_test_class_s1);

[FAR_s1, FRR_s1, EER_s1] = calculate_biometric_metrics(y_pred_s1, y_test_class_s1, 10);

fprintf('\n  ✅ Scenario 1 Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_s1);
fprintf('     EER: %.2f%%\n', EER_s1);
fprintf('     Time: %.2fs\n\n', time_s1);

%% ========================================================================
%  SCENARIO 2: CROSS-DAY (MOST IMPORTANT!)
% =========================================================================

fprintf('===========================================================\n');
fprintf('SCENARIO 2: Cross-Day (Day 1 train → Day 2 test)\n');
fprintf('Most Realistic Evaluation!\n');
fprintf('===========================================================\n\n');

% Use ALL Day 1 for training, ALL Day 2 for testing
X_train_s2 = X_day1;
y_train_s2 = y_day1;
X_test_s2 = X_day2;
y_test_s2 = y_day2;

% Normalize using Day 1 stats only
mu_s2 = mean(X_train_s2, 1);
sigma_s2 = std(X_train_s2, 0, 1);
sigma_s2(sigma_s2 == 0) = 1;

X_train_s2_norm = (X_train_s2 - mu_s2) ./ sigma_s2;
X_test_s2_norm = (X_test_s2 - mu_s2) ./ sigma_s2;

% Transpose
X_train_s2_t = X_train_s2_norm';
X_test_s2_t = X_test_s2_norm';
y_train_s2_oh = full(ind2vec(y_train_s2'));
y_test_s2_oh = full(ind2vec(y_test_s2'));

% Train
fprintf('  Training on Day 1...\n');
net_s2 = patternnet([50, 30], 'trainscg');
net_s2.trainParam.epochs = 500;
net_s2.trainParam.showWindow = false;
net_s2.trainParam.showCommandLine = false;

tic;
[net_s2, ~] = train(net_s2, X_train_s2_t, y_train_s2_oh);
time_s2 = toc;

% Test on Day 2
fprintf('  Testing on Day 2...\n');
y_pred_s2 = net_s2(X_test_s2_t);
y_pred_class_s2 = vec2ind(y_pred_s2);
y_test_class_s2 = vec2ind(y_test_s2_oh);
accuracy_s2 = 100 * sum(y_pred_class_s2 == y_test_class_s2) / length(y_test_class_s2);

[FAR_s2, FRR_s2, EER_s2] = calculate_biometric_metrics(y_pred_s2, y_test_class_s2, 10);

fprintf('\n  ✅ Scenario 2 Results (MOST IMPORTANT):\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_s2);
fprintf('     EER: %.2f%%\n', EER_s2);
fprintf('     Time: %.2fs\n\n', time_s2);

%% ========================================================================
%  SCENARIO 3: COMBINED DAYS
% =========================================================================

fprintf('===========================================================\n');
fprintf('SCENARIO 3: Combined Days (80/20 random split)\n');
fprintf('===========================================================\n\n');

X_combined = [X_day1; X_day2];
y_combined = [y_day1; y_day2];

rng(42);
cv3 = cvpartition(y_combined, 'HoldOut', 0.20);

X_train_s3 = X_combined(training(cv3), :);
X_test_s3 = X_combined(test(cv3), :);
y_train_s3 = y_combined(training(cv3));
y_test_s3 = y_combined(test(cv3));

% Normalize
mu_s3 = mean(X_train_s3, 1);
sigma_s3 = std(X_train_s3, 0, 1);
sigma_s3(sigma_s3 == 0) = 1;

X_train_s3_norm = (X_train_s3 - mu_s3) ./ sigma_s3;
X_test_s3_norm = (X_test_s3 - mu_s3) ./ sigma_s3;

X_train_s3_t = X_train_s3_norm';
X_test_s3_t = X_test_s3_norm';
y_train_s3_oh = full(ind2vec(y_train_s3'));
y_test_s3_oh = full(ind2vec(y_test_s3'));

fprintf('  Training...\n');
net_s3 = patternnet([50, 30], 'trainscg');
net_s3.trainParam.epochs = 500;
net_s3.trainParam.showWindow = false;
net_s3.trainParam.showCommandLine = false;

tic;
[net_s3, ~] = train(net_s3, X_train_s3_t, y_train_s3_oh);
time_s3 = toc;

y_pred_s3 = net_s3(X_test_s3_t);
y_pred_class_s3 = vec2ind(y_pred_s3);
y_test_class_s3 = vec2ind(y_test_s3_oh);
accuracy_s3 = 100 * sum(y_pred_class_s3 == y_test_class_s3) / length(y_test_class_s3);

[FAR_s3, FRR_s3, EER_s3] = calculate_biometric_metrics(y_pred_s3, y_test_class_s3, 10);

fprintf('\n  ✅ Scenario 3 Results:\n');
fprintf('     Accuracy: %.2f%%\n', accuracy_s3);
fprintf('     EER: %.2f%%\n', EER_s3);
fprintf('     Time: %.2fs\n\n', time_s3);

%% ========================================================================
%  COMPARISON TABLE
% =========================================================================

fprintf('===========================================================\n');
fprintf('THREE-SCENARIO COMPARISON\n');
fprintf('===========================================================\n\n');

fprintf('%-40s | %-10s | %-8s\n', 'Scenario', 'Accuracy', 'EER');
fprintf('%-40s | %-10s | %-8s\n', repmat('-',1,40), repmat('-',1,10), repmat('-',1,8));
fprintf('%-40s | %9.2f%% | %7.2f%%\n', 'Scenario 1: Day 1 only', accuracy_s1, EER_s1);
fprintf('%-40s | %9.2f%% | %7.2f%%\n', 'Scenario 2: Cross-Day (Day 1→2)', accuracy_s2, EER_s2);
fprintf('%-40s | %9.2f%% | %7.2f%%\n\n', 'Scenario 3: Combined days', accuracy_s3, EER_s3);

% Save
cross_day_results = struct();
cross_day_results.scenario1 = struct('accuracy', accuracy_s1, 'EER', EER_s1);
cross_day_results.scenario2 = struct('accuracy', accuracy_s2, 'EER', EER_s2);
cross_day_results.scenario3 = struct('accuracy', accuracy_s3, 'EER', EER_s3);
save('results\cross_day_evaluation_results.mat', 'cross_day_results');

fprintf('✅ Results saved!\n');

%% ========================================================================
%  HELPER FUNCTIONS
% =========================================================================

function features = extractFeatures_133(segment)
    % Extract 133 features from 155x6 segment
    % Columns: AccX, AccY, AccZ, GyroX, GyroY, GyroZ
    
    features = [];
    
    % TIME-DOMAIN FEATURES (17 stats × 6 axes = 102 features)
    for col = 1:6
        signal = segment(:, col);
        features = [features, ...
            mean(signal), std(signal), min(signal), max(signal), ...
            rms(signal), median(signal), mad(signal), ...
            skewness(signal), kurtosis(signal), ...
            sum(abs(diff(signal))), iqr(signal), ...
            range(signal), var(signal), ...
            sum(signal < 0) / length(signal), ...
            sum(abs(signal) < 0.1) / length(signal), ...
            max(signal) - min(signal), ...
            sum(abs(signal))];
    end
    
    % FREQUENCY-DOMAIN FEATURES (4 stats × 6 axes = 24 features)
    for col = 1:6
        signal = segment(:, col);
        fft_signal = abs(fft(signal));
        fft_signal = fft_signal(1:floor(length(fft_signal)/2));
        
        features = [features, ...
            max(fft_signal), ...
            mean(fft_signal), ...
            std(fft_signal), ...
            sum(fft_signal.^2)];
    end
    
    % CORRELATION & MAGNITUDE FEATURES (7 features)
    corr_matrix = corrcoef(segment);
    features = [features, ...
        corr_matrix(1,2), corr_matrix(1,3), corr_matrix(2,3), ...
        corr_matrix(4,5), corr_matrix(4,6), corr_matrix(5,6), ...
        sqrt(sum(sum(segment(:,1:3).^2, 2)))];
    
    % Total: 102 + 24 + 7 = 133 features
end

function [FAR, FRR, EER] = calculate_biometric_metrics(y_pred_probs, y_test_class, num_classes)
    all_genuine_scores = [];
    all_impostor_scores = [];
    
    for user_id = 1:num_classes
        user_samples_idx = (y_test_class == user_id);
        
        if sum(user_samples_idx) > 0
            genuine_scores = y_pred_probs(user_id, user_samples_idx);
            all_genuine_scores = [all_genuine_scores, genuine_scores];
            
            impostor_samples_idx = ~user_samples_idx;
            if sum(impostor_samples_idx) > 0
                impostor_preds = y_pred_probs(user_id, impostor_samples_idx);
                all_impostor_scores = [all_impostor_scores, impostor_preds];
            end
        end
    end
    
    thresholds = 0:0.01:1;
    FAR = zeros(size(thresholds));
    FRR = zeros(size(thresholds));
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        FAR(i) = sum(all_impostor_scores >= thresh) / length(all_impostor_scores) * 100;
        FRR(i) = sum(all_genuine_scores < thresh) / length(all_genuine_scores) * 100;
    end
    
    [~, eer_idx] = min(abs(FAR - FRR));
    EER = (FAR(eer_idx) + FRR(eer_idx)) / 2;
end
