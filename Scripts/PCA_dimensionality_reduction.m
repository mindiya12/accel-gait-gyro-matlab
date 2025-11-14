load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

fprintf('\nApplying PCA and testing different component counts...\n');

% --- Remove constant features ---
stdVals = std(featureMatrix, 0, 1);
constantCols = stdVals == 0;
if any(constantCols)
    fprintf('Removing %d constant columns from featureMatrix...\n', sum(constantCols));
    featureMatrix(:, constantCols) = [];
end

% --- Step 1: Validate data ---
if ~exist('featureMatrix', 'var') || ~exist('allLabels', 'var')
    error('Error: featureMatrix or allLabels not found in workspace.');
end
if any(isnan(featureMatrix(:))) || any(isinf(featureMatrix(:)))
    error('Error: featureMatrix contains NaN or Inf values.');
end

% --- Step 2: Standardize features ---
fprintf('Standardizing features...\n');
X_norm = (featureMatrix - mean(featureMatrix, 1)) ./ std(featureMatrix, 0, 1);

% --- Step 3: Perform PCA ---
fprintf('Performing PCA...\n');
[coeff, score, latent, ~, explained] = pca(X_norm);

% --- Step 4: Determine maximum available PCA components ---
maxComponents = size(score, 2);
fprintf('Max available PCA components: %d\n', maxComponents);

% --- Step 5: Define PCA component counts to test ---
requested_components = [50, 75, 100, 120];
pca_components = requested_components(requested_components <= maxComponents);
fprintf('Testing components: %s\n', mat2str(pca_components));

results_pca = zeros(length(pca_components), 1);

% --- Step 6: Loop over PCA component counts ---
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

    % Prepare data for neural network
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    y_test_onehot = full(ind2vec(y_test'));

    % Create neural network
    net = patternnet(30);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;

    % Train network
    fprintf('Training neural network...\n');
    net = train(net, X_train_t, y_train_onehot);

    % Test network
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);

    % Compute accuracy
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    results_pca(i) = accuracy;

    fprintf(' PCA %d components accuracy: %.2f%%\n', k, accuracy * 100);
end

% --- Step 7: Plot results ---
fprintf('\nPlotting results...\n');
figure;
plot(pca_components, results_pca * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of PCA Components');
ylabel('Classification Accuracy (%)');
title('Effect of PCA Dimensionality on Authentication Performance');
grid on;

fprintf('\n PCA testing completed successfully.\n');

