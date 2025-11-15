load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

fprintf('\nApplying PCA and testing different component counts...\n');

% --- Remove constant features ---
stdVals = std(featureMatrix, 0, 1);
constantCols = stdVals == 0;
if any(constantCols)
    fprintf('Removing %d constant columns from featureMatrix...\n', sum(constantCols));
    featureMatrix(:, constantCols) = [];
end

% --- Validate data presence ---
if ~exist('featureMatrix', 'var') || ~exist('allLabels', 'var')
    error('Error: featureMatrix or allLabels not found in workspace.');
end

% --- Check for NaN or Inf values ---
if any(isnan(featureMatrix(:))) || any(isinf(featureMatrix(:)))
    error('Error: featureMatrix contains NaN or Inf values.');
end

% --- Standardize features ---
fprintf('Standardizing features...\n');
X_norm = (featureMatrix - mean(featureMatrix, 1)) ./ std(featureMatrix, 0, 1);

% --- Perform PCA ---
fprintf('Performing PCA...\n');
[coeff, score, latent, ~, explained] = pca(X_norm);

% --- Determine max PCA components ---
maxComponents = size(score, 2);
fprintf('Max available PCA components: %d\n', maxComponents);

% --- Define tested PCA components (restricting to available) ---
requested_components = [50, 75, 100, 120];
pca_components = requested_components(requested_components <= maxComponents);
fprintf('Testing components: %s\n', mat2str(pca_components));

results_pca = zeros(length(pca_components), 1);

% --- Loop over PCA component counts ---
for i = 1:length(pca_components)
    k = pca_components(i);
    fprintf('\nTesting PCA with %d components...\n', k);
    
    X_reduced = score(:, 1:k);
    
    % Split dataset: 80% train, 20% test
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    
    X_train = X_reduced(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_reduced(test(cv), :);
    y_test = allLabels(test(cv));
    
    % Prepare data transpose for neural network
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));
    
    % Create a 2-layer Neural Network: 50 neurons in 1st, 30 neurons in 2nd
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    
    fprintf('Training neural network...\n');
    net = train(net, X_train_t, y_train_onehot);
    
    % Test the network
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    
    % Calculate classification accuracy
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    results_pca(i) = accuracy;
    fprintf(' PCA %d components accuracy: %.2f%%\n', k, accuracy * 100);
end

% --- Plot results ---
fprintf('\nPlotting results...\n');
figure;
plot(pca_components, results_pca * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of PCA Components');
ylabel('Classification Accuracy (%)');
title('Effect of PCA Dimensionality on Authentication Performance');
grid on;

fprintf('\nPCA testing with 2-layer neural network completed successfully.\n');
