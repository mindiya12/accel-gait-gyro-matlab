load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

% Top 20 important features indices from your analysis
top20_features = [32    73    13    15    17    19    22    26    56    69    71   123   132     2     6     7   12    18    48    61];

fprintf('Training and evaluating model using TOP 20 features only...\n');

% Extract top 20 features
X_top20 = featureMatrix(:, top20_features);

% Normalize features with z-score
mu = mean(X_top20);
sigma = std(X_top20);
sigma(sigma == 0) = 1; % Avoid divide by zero
X_norm = (X_top20 - mu) ./ sigma;

% Split into training and testing sets (80/20 random split)
cv = cvpartition(allLabels, 'HoldOut', 0.2);

X_train = X_norm(training(cv), :);
y_train = allLabels(training(cv));
X_test = X_norm(test(cv), :);
y_test = allLabels(test(cv));

% Prepare data for MATLAB Neural Network
X_train_t = X_train';
y_train_onehot = full(ind2vec(y_train'));
X_test_t = X_test';
y_test_onehot = full(ind2vec(y_test'));

% Create and configure feedforward neural network
hiddenLayerSize = [50 30];  % Use your best architecture or baseline
net = patternnet(hiddenLayerSize);
net.trainParam.epochs = 500;
net.trainParam.showWindow = false; % Disable GUI during training

% Train network
fprintf('Training network...\n');
net = train(net, X_train_t, y_train_onehot);

% Predict on test set
y_pred = net(X_test_t);
y_pred_class = vec2ind(y_pred);

% Calculate accuracy
accuracy = sum(y_pred_class' == y_test) / length(y_test);
fprintf('Accuracy with top 20 features: %.2f%%\n', accuracy * 100);

% Save results for report
save('results/model_top20_features.mat', 'net', 'top20_features', 'accuracy', 'mu', 'sigma');

% Optional: plot confusion matrix
figure;
plotconfusion(y_test_onehot, y_pred);
title('Confusion Matrix - Top 20 Features');
