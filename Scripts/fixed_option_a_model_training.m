% =========================================================================
% OPTION A: CROSS-USER AUTHENTICATION (IMPROVED VERSION)
% Trains on Users 1-8, Tests authentication on ALL users
% =========================================================================
clear; clc;

fprintf('==========================================================\n');
fprintf('OPTION A: CROSS-USER AUTHENTICATION (IMPROVED)\n');
fprintf('==========================================================\n');

% --- Load preprocessed data ---
load('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat');

fprintf('\n[STEP 1] Data loaded\n');
fprintf('  Training users (1-8): %d samples\n', size(X_train_A_norm, 1));
fprintf('  Test users (9-10): %d samples\n', size(X_test_A_norm, 1));

% --- Prepare training data (Users 1-8 only) ---
X_train = X_train_A_norm';  % Transpose for neural network
y_train = y_train_A;
y_train_onehot = full(ind2vec(y_train'));  % One-hot encoding

% --- Configure Neural Network ---
fprintf('\n[STEP 2] Configuring Neural Network...\n');
hiddenLayerSize = [64, 32];  % Slightly larger network
trainFcn = 'trainscg';

net_A = patternnet(hiddenLayerSize, trainFcn);
net_A.trainParam.epochs = 500;
net_A.trainParam.showWindow = true;
net_A.trainParam.showCommandLine = false;

% Data division
net_A.divideParam.trainRatio = 0.70;
net_A.divideParam.valRatio = 0.15;
net_A.divideParam.testRatio = 0.15;

fprintf('  Architecture: %d -> [64, 32] -> 8 users\n', size(X_train, 1));

% --- Train the network ---
fprintf('\n[STEP 3] Training on Users 1-8...\n');
[net_A, tr_A] = train(net_A, X_train, y_train_onehot);

% --- Evaluate on training users (Users 1-8) - Genuine attempts ---
fprintf('\n[STEP 4] Testing on enrolled users (Users 1-8)...\n');
outputs_train = net_A(X_train);
max_conf_genuine = max(outputs_train, [], 1);  % Max confidence per sample

fprintf('  Genuine samples: %d\n', length(max_conf_genuine));
fprintf('  Mean confidence: %.4f\n', mean(max_conf_genuine));
fprintf('  Std confidence: %.4f\n', std(max_conf_genuine));

% --- Test on IMPOSTOR users (Users 9-10) ---
fprintf('\n[STEP 5] Testing on impostor users (Users 9-10)...\n');
X_test = X_test_A_norm';
outputs_impostor = net_A(X_test);
max_conf_impostor = max(outputs_impostor, [], 1);  % Max confidence per sample

fprintf('  Impostor samples: %d\n', length(max_conf_impostor));
fprintf('  Mean confidence: %.4f\n', mean(max_conf_impostor));
fprintf('  Std confidence: %.4f\n', std(max_conf_impostor));

% --- Plot confidence distributions ---
figure('Name', 'Confidence Score Distributions');
subplot(2,1,1);
histogram(max_conf_genuine, 50, 'FaceColor', 'green', 'FaceAlpha', 0.6);
xlabel('Max Confidence Score');
ylabel('Count');
title('Genuine Users (1-8) - Should have HIGH confidence');
grid on;

subplot(2,1,2);
histogram(max_conf_impostor, 50, 'FaceColor', 'red', 'FaceAlpha', 0.6);
xlabel('Max Confidence Score');
ylabel('Count');
title('Impostor Users (9-10) - Should have LOW confidence');
grid on;

% --- Calculate FAR and FRR at different thresholds ---
fprintf('\n[STEP 6] Calculating FAR and FRR across thresholds...\n');

thresholds = 0:0.01:1;  % Test 101 different thresholds
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));

for i = 1:length(thresholds)
    thresh = thresholds(i);
    
    % FRR: Genuine users rejected (should be accepted)
    genuine_rejected = sum(max_conf_genuine < thresh);
    FRR(i) = genuine_rejected / length(max_conf_genuine);
    
    % FAR: Impostor users accepted (should be rejected)
    impostor_accepted = sum(max_conf_impostor >= thresh);
    FAR(i) = impostor_accepted / length(max_conf_impostor);
end

% --- Find Equal Error Rate (EER) ---
[~, eer_idx] = min(abs(FAR - FRR));
EER = FAR(eer_idx);
EER_threshold = thresholds(eer_idx);

fprintf('  Equal Error Rate (EER): %.2f%% at threshold=%.2f\n', EER*100, EER_threshold);
fprintf('  At EER threshold: FAR=%.2f%%, FRR=%.2f%%\n', FAR(eer_idx)*100, FRR(eer_idx)*100);

% --- Plot ROC-like curve (FAR vs FRR) ---
figure('Name', 'FAR vs FRR Curve');
plot(thresholds, FAR*100, 'r-', 'LineWidth', 2); hold on;
plot(thresholds, FRR*100, 'b-', 'LineWidth', 2);
plot(EER_threshold, EER*100, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'yellow');
xlabel('Threshold');
ylabel('Error Rate (%)');
title('FAR vs FRR - Option A (Cross-User Authentication)');
legend('FAR (False Accept)', 'FRR (False Reject)', sprintf('EER=%.2f%% @ %.2f', EER*100, EER_threshold), 'Location', 'best');
grid on;

% --- Test at common thresholds ---
fprintf('\n[STEP 7] Performance at common thresholds:\n');
test_thresholds = [0.5, 0.7, 0.9];
for thresh = test_thresholds
    idx = find(thresholds == thresh);
    if ~isempty(idx)
        fprintf('  Threshold %.2f: FAR=%.2f%%, FRR=%.2f%%\n', ...
            thresh, FAR(idx)*100, FRR(idx)*100);
    end
end

% --- Calculate optimal threshold (balanced security/usability) ---
% Option 1: Minimize total error
total_error = FAR + FRR;
[~, opt_idx] = min(total_error);
opt_threshold = thresholds(opt_idx);

fprintf('\n[STEP 8] Optimal threshold (min total error):\n');
fprintf('  Threshold: %.2f\n', opt_threshold);
fprintf('  FAR: %.2f%%\n', FAR(opt_idx)*100);
fprintf('  FRR: %.2f%%\n', FRR(opt_idx)*100);
fprintf('  Total Error: %.2f%%\n', total_error(opt_idx)*100);

% --- Save results ---
fprintf('\n[STEP 9] Saving results...\n');
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\trained_model_OptionA_improved.mat', ...
    'net_A', 'tr_A', 'hiddenLayerSize', 'EER', 'EER_threshold', ...
    'FAR', 'FRR', 'thresholds', 'max_conf_genuine', 'max_conf_impostor');

fprintf('  ✅ Model saved: trained_model_OptionA_improved.mat\n');

% --- Final Summary ---
fprintf('\n==========================================================\n');
fprintf('OPTION A AUTHENTICATION RESULTS\n');
fprintf('==========================================================\n');
fprintf('Training: Users 1-8\n');
fprintf('Testing: Users 9-10 (impostors)\n');
fprintf('Equal Error Rate (EER): %.2f%%\n', EER*100);
fprintf('EER Threshold: %.2f\n', EER_threshold);
fprintf('Optimal Threshold: %.2f (FAR=%.2f%%, FRR=%.2f%%)\n', ...
    opt_threshold, FAR(opt_idx)*100, FRR(opt_idx)*100);
fprintf('==========================================================\n');

fprintf('\nInterpretation:\n');
fprintf('- Lower EER = Better authentication system\n');
fprintf('- FAR: How often impostors are accepted (security risk)\n');
fprintf('- FRR: How often genuine users are rejected (usability issue)\n');
fprintf('- Choose threshold based on your security requirements\n');
