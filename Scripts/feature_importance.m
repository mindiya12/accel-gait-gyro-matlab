load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');
fprintf('\nStarting sequential feature importance...\n');

% Use baseline accuracy from your previous feature subset testing with all features
baseAcc = 0.992; % Assuming 'results' variable exists in workspace with final accuracy of all-feature model

numFeat = size(featureMatrix, 2);
featImpact = zeros(numFeat, 1);

for f = 1:numFeat
    fprintf('Removing feature %d / %d\n', f, numFeat);
    cols = setdiff(1:numFeat, f);  % Exclude current feature
    
    X_subset = featureMatrix(:, cols);

    % Normalize features (z-score)
    mu = mean(X_subset, 1);
    sigma = std(X_subset, [], 1);
    sigma(sigma==0) = 1;  % Prevent division by zero
    X_norm = (X_subset - mu) ./ sigma;
    
    % Split dataset into 80% train and 20% test using cvpartition
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
    
    % Create and train 2-layer neural network with 50 and 30 neurons
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    
    % Prepare data for neural net format
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));
    
    % Train the network
    net = train(net, X_train_t, y_train_onehot);
    
    % Predict on test set
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    
    % Calculate accuracy and impact on accuracy
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    featImpact(f) = baseAcc - accuracy;
    
    fprintf(' Removed feature %d, accuracy drop: %.4f\n', f, featImpact(f));
end

% Identify top 20 most important features (largest accuracy drops)
[~, idxs] = sort(featImpact, 'descend');
top20 = idxs(1:20);
fprintf('\nTop 20 most important features:\n');
disp(top20');

% Plot feature importance for top 20 features
figure;
bar(featImpact(idxs(1:20)));
xlabel('Feature Index (Top 20)');
ylabel('Accuracy Drop on Removal');
title('Top 20 Feature Importance (Sequential Removal Impact)');
grid on;
