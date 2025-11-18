% PCA DIMENSION REDUCTION EXPERIMENT
% Per-User EER/FAR/FRR evaluation
clear; clc;
load('D:\src\development\accel-gait-gyro-matlab\results\extracted_features.mat');

fprintf('\nApplying PCA and testing different component counts...\n');

% Remove constant features
stdVals = std(featureMatrix, 0, 1);
constantCols = stdVals == 0;
if any(constantCols)
    fprintf('Removing %d constant columns from featureMatrix...\n', sum(constantCols));
    featureMatrix(:, constantCols) = [];
end

if any(isnan(featureMatrix(:))) || any(isinf(featureMatrix(:)))
    error('Feature matrix contains NaN or Inf values.');
end

% Standardize features 
fprintf('Standardizing features...\n');
X_norm = (featureMatrix - mean(featureMatrix, 1)) ./ std(featureMatrix, 0, 1);

% Perform PCA
fprintf('Performing PCA...\n');
[coeff, score, latent, ~, explained] = pca(X_norm);

% Determine tested PCA component counts 
maxComponents = size(score, 2);
requested_components = [50, 75, 100, 120];
pca_components = requested_components(requested_components <= maxComponents);
fprintf('Testing components: %s\n', mat2str(pca_components));

numUsers = 10;
thresholds = 0:0.01:1;

mean_EERs = zeros(length(pca_components), 1);
std_EERs = zeros(length(pca_components), 1);
mean_FARs = zeros(length(pca_components), 1);
std_FARs = zeros(length(pca_components), 1);
mean_FRRs = zeros(length(pca_components), 1);
std_FRRs = zeros(length(pca_components), 1);

resultsFolder = 'D:\src\development\accel-gait-gyro-matlab\results\figures_member3\pca_exp';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

for i = 1:length(pca_components)
    k = pca_components(i);
    fprintf('\nTesting PCA with %d components...\n', k);
   
    X_reduced = score(:, 1:k);
   
    % Split data 80/20
    rng(4000 + k);
    cv = cvpartition(allLabels, 'HoldOut', 0.2);
    X_train = X_reduced(training(cv), :);
    y_train = allLabels(training(cv));
    X_test = X_reduced(test(cv), :);
    y_test = allLabels(test(cv));
   
    % NN setup
    net = patternnet([50 30]);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 200;
    net.divideMode = 'none';
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
   
    fprintf('Training neural network...\n');
    net = train(net, X_train_t, y_train_onehot);
   
    y_pred = net(X_test_t);
    y_test_class = y_test(:)';
   
    % Per-user EER/FAR/FRR 
    fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(i), std_EERs(i));
    fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', mean_FARs(i), std_FARs(i));
    fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', mean_FRRs(i), std_FRRs(i));
end

% Report plot: EER vs. PCA components
figure('Visible','off', 'Position', [120,100,900,500]);
errorbar(pca_components, mean_EERs, std_EERs, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.1 0.5 0.8]);
xlabel('Number of PCA Components', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate (EER) (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Impact of PCA Dimensionality on Authentication Security (EER)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);
for i = 1:length(pca_components)
    text(pca_components(i), mean_EERs(i) + std_EERs(i) + 0.3, ...
        sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
saveas(gcf, fullfile(resultsFolder, 'PCA_EER_vs_Components.png'));
close;

% Save summary table
summary_table = table(pca_components(:), mean_EERs, std_EERs, mean_FARs, std_FARs, mean_FRRs, std_FRRs, ...
    'VariableNames', {'PCA_Components','Mean_EER','Std_EER','Mean_FAR','Std_FAR','Mean_FRR','Std_FRR'});
writetable(summary_table, fullfile(resultsFolder, 'PCA_EER_Summary.csv'));

fprintf('\nPCA EXPERIMENTS COMPLETE!\n');
fprintf('Results saved to: %s\n', resultsFolder);
fprintf('Main plot: PCA_EER_vs_Components.png\n');
fprintf('Summary table: PCA_EER_Summary.csv\n\n');