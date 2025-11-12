% =========================================================================
% NEURAL NETWORK TRAINING - OPTION A AND OPTION B
% File: train_neural_networks.m
% =========================================================================
clear; clc;

fprintf('==========================================================\n');
fprintf('NEURAL NETWORK TRAINING FOR USER AUTHENTICATION\n');
fprintf('==========================================================\n');

% Load preprocessed data
fprintf('\n[STEP 1] Loading preprocessed data...\n');
load('D:\src\development\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat');

fprintf('  Option B loaded: %d train, %d test samples\n', ...
    size(X_train_B_norm, 1), size(X_test_B_norm, 1));


% =========================================================================
% OPTION B: TRAIN NEURAL NETWORK (Within-User Authentication)
% =========================================================================
fprintf('\n==========================================================\n');
fprintf('TRAINING OPTION B: Within-User Authentication\n');
fprintf('Random 80/20 split across all users\n');
fprintf('==========================================================\n');

% Prepare data for patternnet
X_train_B_t = X_train_B_norm';  % 133 features × 1079 samples
X_test_B_t = X_test_B_norm';    % 133 features × 270 samples

% Convert labels to one-hot encoding (10 classes for all users)
y_train_B_onehot = full(ind2vec(y_train_B'));  % 10 classes × 1079 samples
y_test_B_onehot = full(ind2vec(y_test_B'));    % 10 classes × 270 samples

fprintf('\n[STEP 2] Configuring Neural Network for Option B...\n');

% Network architecture (same as Option A for fair comparison)
hiddenLayerSize = [50 30];
trainFcn = 'trainscg';

% Create pattern recognition network
net_B = patternnet(hiddenLayerSize, trainFcn);

% Configure training parameters
net_B.trainParam.epochs = 500;
net_B.trainParam.goal = 1e-5;
net_B.trainParam.showWindow = true;
net_B.trainParam.showCommandLine = false;

% Data division
net_B.divideParam.trainRatio = 0.70;
net_B.divideParam.valRatio = 0.15;
net_B.divideParam.testRatio = 0.15;

fprintf('  Architecture: %d inputs -> [50, 30] hidden -> %d outputs\n', ...
    size(X_train_B_t, 1), size(y_train_B_onehot, 1));
fprintf('  Training function: %s\n', trainFcn);
fprintf('  Max epochs: %d\n', net_B.trainParam.epochs);

fprintf('\n[STEP 3] Training Option B Neural Network...\n');
fprintf('  (This may take 1-3 minutes depending on your system)\n\n');

% Train the network
[net_B, tr_B] = train(net_B, X_train_B_t, y_train_B_onehot);

fprintf('  Training complete!\n');
fprintf('  Final performance: %.6f\n', tr_B.best_perf);

% =========================================================================
% EVALUATE OPTION B
% =========================================================================
fprintf('\n[STEP 4] Evaluating Option B on Test Set...\n');

% Predict on test set
y_pred_B = net_B(X_test_B_t);
y_pred_B_class = vec2ind(y_pred_B);  % Convert from one-hot to class indices
y_test_B_class = vec2ind(y_test_B_onehot);

% Calculate accuracy
correct_B = sum(y_pred_B_class == y_test_B_class);
accuracy_B = 100 * correct_B / length(y_test_B_class);

fprintf('  Test accuracy: %.2f%% (%d/%d correct)\n', ...
    accuracy_B, correct_B, length(y_test_B_class));

% Calculate performance metrics
test_perf_B = perform(net_B, y_test_B_onehot, y_pred_B);
fprintf('  Test performance (MSE): %.6f\n', test_perf_B);

% Plot confusion matrix
figure('Name', 'Option B: Confusion Matrix');
plotconfusion(y_train_B_onehot, net_B(X_train_B_t), 'Training', ...
              y_test_B_onehot, y_pred_B, 'Testing');

% Detailed confusion matrix
C_B = confusionmat(y_test_B_class, y_pred_B_class);
fprintf('\n  Confusion Matrix (rows=true, cols=predicted):\n');
disp(C_B);

% Calculate per-class accuracy
fprintf('\n  Per-User Accuracy:\n');
for u = 1:10
    user_idx = (y_test_B_class == u);
    if sum(user_idx) > 0
        user_correct = sum(y_pred_B_class(user_idx) == u);
        user_total = sum(user_idx);
        user_acc = 100 * user_correct / user_total;
        fprintf('    User %d: %.1f%% (%d/%d)\n', u, user_acc, user_correct, user_total);
    end
end

% =========================================================================
% SAVE TRAINED MODEL B
% =========================================================================
fprintf('\n[STEP 5] Saving trained model for Option B...\n');
save('D:\src\development\accel-gait-gyro-matlab\results/trained_model_OptionB.mat', ...
    'net_B', 'tr_B', 'hiddenLayerSize', 'trainFcn');
fprintf('  Saved: trained_model_OptionB.mat\n');

fprintf('\n==========================================================\n');
fprintf('OPTION B TRAINING COMPLETE\n');
fprintf('==========================================================\n');

