% ============================================================
% COMPARE OVERLAPPING VS NON-OVERLAPPING PERFORMANCE
% Uses DUPLICATE-FREE datasets for fair comparison
% ============================================================

clear; clc;

fprintf('=== COMPARING OVERLAPPING WINDOW CONFIGURATIONS (NO DUPLICATES) ===\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Load all three CLEANED datasets ---
configs = {
    'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

% Storage for results
results = struct();

fprintf('Loading cleaned datasets...\n');
for i = 1:size(configs, 1)
    filePath = fullfile(resultsFolder, configs{i, 1});
    if exist(filePath, 'file')
        load(filePath);
        fprintf('  %s: %d samples x %d features\n', configs{i, 2}, ...
                size(featureMatrix, 1), size(featureMatrix, 2));
    else
        error('Cleaned dataset not found: %s', configs{i, 1});
    end
end

fprintf('\n');

% --- Evaluation function (SAME AS BEFORE) ---
function [accuracy, eer, far, frr, trainTime, confMat] = evaluateConfiguration(X, y, configName)
    fprintf('--- Evaluating: %s ---\n', configName);
    fprintf('Dataset: %d samples, %d users\n', size(X, 1), length(unique(y)));
    
    % Normalize features (z-score)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;
    X_norm = (X - mu) ./ sigma;
    
    % 80/20 train-test split (subject-independent)
    rng(42); % Fixed seed for reproducibility
    cv = cvpartition(y, 'HoldOut', 0.2);
    
    X_train = X_norm(training(cv), :);
    y_train = y(training(cv));
    X_test = X_norm(test(cv), :);
    y_test = y(test(cv));
    
    fprintf('  Train set: %d samples\n', size(X_train, 1));
    fprintf('  Test set: %d samples\n', size(X_test, 1));
    
    % Train neural network - SAME ARCHITECTURE FOR ALL
    net = patternnet([50, 30]); % Your baseline architecture
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 500;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;
    
    % Prepare data
    X_train_t = X_train';
    y_train_onehot = full(ind2vec(y_train'));
    X_test_t = X_test';
    
    % Train and time it
    fprintf('  Training neural network...\n');
    tic;
    net = train(net, X_train_t, y_train_onehot);
    trainTime = toc;
    fprintf('  Training completed in %.2f seconds\n', trainTime);
    
    % Predict
    y_pred = net(X_test_t);
    y_pred_class = vec2ind(y_pred);
    
    % Calculate metrics
    accuracy = sum(y_pred_class' == y_test) / length(y_test);
    
    % Confusion matrix
    confMat = confusionmat(y_test, y_pred_class');
    
    % FAR and FRR
    totalSamples = sum(confMat(:));
    correctPredictions = sum(diag(confMat));
    
    far = (totalSamples - correctPredictions) / totalSamples;
    frr = 1 - accuracy;
    eer = (far + frr) / 2;
    
    % Display results
    fprintf('  Results:\n');
    fprintf('    Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('    EER: %.4f\n', eer);
    fprintf('    FAR: %.4f\n', far);
    fprintf('    FRR: %.4f\n', frr);
    fprintf('    Training time: %.2f seconds\n\n', trainTime);
end

% --- Evaluate all three configurations ---
for configIdx = 1:size(configs, 1)
    featureFile = configs{configIdx, 1};
    configName = configs{configIdx, 2};
    
    % Load cleaned feature matrix
    load(fullfile(resultsFolder, featureFile));
    
    % Evaluate
    [acc, eer_val, far_val, frr_val, time, cm] = ...
        evaluateConfiguration(featureMatrix, allLabels, configName);
    
    % Store results
    results(configIdx).name = configName;
    results(configIdx).numSamples = size(featureMatrix, 1);
    results(configIdx).accuracy = acc;
    results(configIdx).eer = eer_val;
    results(configIdx).far = far_val;
    results(configIdx).frr = frr_val;
    results(configIdx).trainTime = time;
    results(configIdx).confMat = cm;
end

% --- Create comparison table ---
fprintf('==========================================================\n');
fprintf('COMPARISON SUMMARY (DUPLICATE-FREE DATASETS)\n');
fprintf('==========================================================\n\n');

Method = {results.name}';
NumSamples = [results.numSamples]';
Accuracy = [results.accuracy]' * 100;
EER = [results.eer]';
FAR = [results.far]';
FRR = [results.frr]';
TrainingTime = [results.trainTime]';

comparisonTable = table(Method, NumSamples, Accuracy, EER, FAR, FRR, TrainingTime);
disp(comparisonTable);

% Calculate improvements
fprintf('\n--- Performance Differences from Baseline ---\n');
baselineAcc = Accuracy(1);
for i = 2:length(Accuracy)
    diff = Accuracy(i) - baselineAcc;
    fprintf('%s: ', Method{i});
    if diff > 0
        fprintf('+%.2f%% improvement\n', diff);
    elseif diff < 0
        fprintf('%.2f%% decrease\n', diff);
    else
        fprintf('no change\n');
    end
end

% --- Save results ---
save(fullfile(resultsFolder, 'overlap_comparison_results_cleaned.mat'), ...
     'results', 'comparisonTable');

% --- Visualizations (SAME AS BEFORE) ---
figure('Position', [100, 100, 1400, 500]);

subplot(1, 4, 1);
bar(Accuracy);
set(gca, 'xticklabel', {'0%', '25%', '50%'});
ylabel('Accuracy (%)');
title('Classification Accuracy');
ylim([min(Accuracy)-5, 100]);
grid on;
for i = 1:length(Accuracy)
    text(i, Accuracy(i)+1, sprintf('%.1f%%', Accuracy(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end

subplot(1, 4, 2);
bar(EER * 100);
set(gca, 'xticklabel', {'0%', '25%', '50%'});
ylabel('EER (%)');
title('Equal Error Rate');
grid on;

subplot(1, 4, 3);
bar(NumSamples);
set(gca, 'xticklabel', {'0%', '25%', '50%'});
ylabel('Number of Samples');
title('Dataset Size (After Deduplication)');
grid on;
for i = 1:length(NumSamples)
    text(i, NumSamples(i)+50, sprintf('%d', NumSamples(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end

subplot(1, 4, 4);
bar(TrainingTime);
set(gca, 'xticklabel', {'0%', '25%', '50%'});
ylabel('Time (seconds)');
title('Training Time');
grid on;

sgtitle('Overlapping vs Non-Overlapping (Duplicate-Free Comparison)');

saveas(gcf, fullfile(resultsFolder, 'overlap_comparison_cleaned.png'));
saveas(gcf, fullfile(resultsFolder, 'overlap_comparison_cleaned.fig'));

fprintf('\n==========================================================\n');
fprintf('Comparison complete!\n');
fprintf('Results saved to: overlap_comparison_results_cleaned.mat\n');
fprintf('Visualization saved to: overlap_comparison_cleaned.png\n');
fprintf('==========================================================\n');
