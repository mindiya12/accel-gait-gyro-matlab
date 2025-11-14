load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

fprintf('\nTesting different normalization methods...\n');

norm_methods = {'zscore', 'minmax', 'robust', 'none'};
results_norm = zeros(length(norm_methods), 1);

for i = 1:length(norm_methods)
    method = norm_methods{i};
    fprintf('Method: %s\n', method);
    
    X_norm = featureMatrix;
    
    switch method
        case 'zscore'
            mu = mean(X_norm);
            sigma = std(X_norm);
            sigma(sigma==0) = 1;
            X_norm = (X_norm - mu) ./ sigma;
        case 'minmax'
            X_min = min(X_norm);
            X_max = max(X_norm);
            range = X_max - X_min;
            range(range==0) = 1;
            X_norm = (X_norm - X_min) ./ range;
        case 'robust'
            med = median(X_norm);
            iqr_val = prctile(X_norm, 75) - prctile(X_norm, 25);
            iqr_val(iqr_val==0) = 1;
            X_norm = (X_norm - med) ./ iqr_val;
        case 'none'
            % No normalization
    end
    
    % Split data
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_norm(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = allLabels(test(cv));
    
    % Train and evaluate as before
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
    results_norm(i) = accuracy;
    fprintf(' Normalization %s accuracy: %.2f%%\n', method, accuracy*100);
end

% Plot Normalization results
figure;
bar(results_norm*100);
set(gca, 'xticklabel', norm_methods);
ylabel('Accuracy (%)');
title('Impact of Normalization Methods on Performance');
grid on;
