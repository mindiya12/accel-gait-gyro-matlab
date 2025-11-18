% Load preprocessed normalized features
load('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results/normalized_splits_FINAL.mat');

% --- Prepare training data (Users 1-8) ---
X_train = X_train_A_norm;         % Features normalized
y_train = y_train_A;             % Labels
X_train_t = X_train';            % Transpose for neural network (features x samples)
y_train_onehot = full(ind2vec(y_train')); % One-hot encode labels for training

% --- Prepare test data (Users 9-10) ---
X_test = X_test_A_norm;
y_test = y_test_A;
X_test_t = X_test';

% --- Define and configure neural network ---
hiddenLayerSize = [50, 30];
netA = patternnet(hiddenLayerSize, 'trainscg');
netA.trainParam.epochs = 500;
netA.divideParam.trainRatio = 0.7;
netA.divideParam.valRatio = 0.15;
netA.divideParam.testRatio = 0.15;

% --- Train neural network on enrolled users (1-8) ---
[netA, trA] = train(netA, X_train_t, y_train_onehot);

% --- Evaluate training accuracy ---
outputs_train = netA(X_train_t);
train_pred_class = vec2ind(outputs_train);
train_true_class = y_train';
train_acc = mean(train_pred_class == train_true_class) * 100;
fprintf('Training accuracy (Users 1–8): %.2f%%\n', train_acc);

% --- Authenticate test users (9-10) ---
outputs_test = netA(X_test_t);          % Network outputs (8 x N)
max_scores_test = max(outputs_test, [], 1); % Max confidence scores

% Set confidence threshold for authentication
auth_threshold = 0.8;
auth_results = max_scores_test >= auth_threshold;
num_authenticated = sum(auth_results);
num_test_samples = length(auth_results);
fprintf('Authenticated as enrolled users (threshold %.2f): %d out of %d test samples (Users 9-10)\n', auth_threshold, num_authenticated, num_test_samples);

% --- Calculate False Acceptance Rate (FAR) ---
FAR = (num_authenticated / num_test_samples) * 100;
fprintf('False Acceptance Rate (FAR) on Users 9-10: %.2f%%\n', FAR);

% --- Visualize confidence scores for test samples ---
figure;
histogram(max_scores_test, 20);
xlabel('Max Prediction Confidence');
ylabel('Count');
title('Confidence Scores for Test Users (9–10) - Cross User Authentication');

% --- Save model ---
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results/trained_model_OptionA.mat', 'netA', 'trA', 'hiddenLayerSize');
fprintf('Model saved to trained_model_OptionA.mat\n');
