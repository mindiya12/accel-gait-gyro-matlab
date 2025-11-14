load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');
fprintf('\nStarting sequential feature importance...\n');
baseAcc = results(end); % Accuracy using all features from above

numFeat = size(featureMatrix, 2);
featImpact = zeros(numFeat, 1);

for f = 1:numFeat
    fprintf('Removing feature %d / %d\n', f, numFeat);
    cols = setdiff(1:numFeat, f);  % Exclude current feature
    
    X_subset = featureMatrix(:, cols);
    % Normalize
    mu = mean(X_subset, 1);
    sigma = std(X_subset, [], 1);
    sigma(sigma==0) = 1;
    X_norm = (X_subset - mu) ./ sigma;
    
    % Use same training/test splitting as baseline for fairness
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
    
    % Train small network
    net = patternnet(30);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));
    
    net = train(net, X_train_t, y_train_onehot);
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    featImpact(f) = baseAcc - accuracy;  % Impact on accuracy
    
    fprintf(' Removed feature %d, accuracy drop: %.4f\n', f, featImpact(f));
end

% Identify top 20 most important features (largest drops)
[~, idxs] = sort(featImpact, 'descend');
top20 = idxs(1:20);

fprintf('\nTop 20 most important features:\n');
disp(top20');

% Plot feature importance
figure;
bar(featImpact(idxs(1:20)));
xlabel('Feature Index');
ylabel('Accuracy Drop');
title('Top 20 Feature Importance (Removal Impact)');
grid on;
