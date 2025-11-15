load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

feature_sets = {
    1:102,            % Time-domain features only
    103:126,          % Frequency-domain features only
    127:133,          % Correlation + magnitude features only
    [1:50, 127:133],  % Selected time-domain + correlation + magnitude
    1:133             % All features - baseline
};

numSets = length(feature_sets);
results = zeros(numSets, 1);  % To store accuracy for each subset

for i = 1:numSets
    fprintf('\nTesting Feature Subset %d...\n', i);
    cols = feature_sets{i};
    
    % Extract subset of features
    X_subset = featureMatrix(:, cols);
    
    % Normalize subset (z-score)
    mu = mean(X_subset, 1);
    sigma = std(X_subset, [], 1);
    sigma(sigma==0) = 1;  % Avoid division by zero
    X_norm = (X_subset - mu) ./ sigma;
    
    % Split data (80/20 random split)
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
    
    % Create feedforward neural network with two hidden layers: 50 and 30 neurons
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;  % Hide training GUI
    net.trainParam.epochs = 200;
    
    % Prepare data for MATLAB NN
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));
    
    % Train the network
    net = train(net, X_train_t, y_train_onehot);
    
    % Predict on test set
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    
    % Calculate accuracy
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    fprintf(' Feature Subset %d Accuracy = %.2f%%\n', i, accuracy*100);
    
    results(i) = accuracy;
end

% Plot results
figure;
bar(results*100);
xlabel('Feature Subset');
ylabel('Accuracy (%)');
title('Classification Accuracy for Different Feature Subsets');
xticklabels({'Time domain', 'Frequency domain', 'Corr+Mag', 'Time+Corr', 'All features'});
grid on;
