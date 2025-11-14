% --- VARYING TRAIN/TEST SPLITS FOR NN AUTHENTICATION ---
clear; clc;

% Load the extracted features and labels
data = load('C:\Users\ASUS\Desktop\MATLAB\accel-gait-gyro-matlab\results\extracted_features.mat');
X = data.featureMatrix;          % Features: N × D
Y = data.allLabels;              % Labels: N × 1

split_ratios = [0.60 0.70 0.75 0.80 0.85 0.90];
EER_results = zeros(length(split_ratios), 1);
FAR_results = zeros(length(split_ratios), 1);
FRR_results = zeros(length(split_ratios), 1);

for idx = 1:length(split_ratios)
    ratio = split_ratios(idx);
    % ======== Shuffle and Split =======
    N = size(X,1);
    rng(42 + idx); % Reproducibility
    order = randperm(N);
    n_train = round(N * ratio);
    train_idx = order(1:n_train);
    test_idx = order(n_train+1:end);
    
    X_train = X(train_idx, :); Y_train = Y(train_idx);
    X_test  = X(test_idx, :);  Y_test  = Y(test_idx);
    
    % ======== Normalize (Z-score with training stats) =======
    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1); sigma(sigma < 1e-8) = 1;
    X_train_norm = (X_train - mu) ./ sigma;
    X_test_norm = (X_test - mu) ./ sigma;
    
    % ======== Prepare For Neural Net (patternnet) =======
    X_train_t = X_train_norm';
    X_test_t  = X_test_norm';
    y_train_onehot = full(ind2vec(Y_train'));
    y_test_onehot  = full(ind2vec(Y_test'));
    
    % ======== Train Neural Network =======
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.trainParam.showWindow = false;
    
    [net, tr] = train(net, X_train_t, y_train_onehot);
    
    % ======== Test and Evaluate =======
    outputs_train = net(X_train_t);
    outputs_test  = net(X_test_t);
    maxConf_train = max(outputs_train, [], 1);
    maxConf_test = max(outputs_test, [], 1);
    
    % --- Compute EER (Equal Error Rate) ---
    thresholds = 0:0.01:1;
    fr_train = zeros(size(thresholds));
    fa_test = zeros(size(thresholds));
    for t = 1:length(thresholds)
        th = thresholds(t);
        % FRR: fraction of genuine samples rejected (train set)
        fr_train(t) = sum(maxConf_train < th) / length(maxConf_train);
        % FAR: fraction of impostor samples accepted (test set)
        fa_test(t)  = sum(maxConf_test >= th) / length(maxConf_test);
    end
    % EER: Threshold where FAR and FRR are closest
    [~, eer_idx] = min(abs(fr_train - fa_test));
    EER = mean([fr_train(eer_idx), fa_test(eer_idx)]);    
    
    EER_results(idx) = EER*100;
    FAR_results(idx) = fa_test(eer_idx)*100;
    FRR_results(idx) = fr_train(eer_idx)*100;
    
    % Print each result
    fprintf('Split %.0f%% train: EER = %.2f%% (FAR = %.2f%%, FRR = %.2f%%)\n', ...
        ratio*100, EER*100, fa_test(eer_idx)*100, fr_train(eer_idx)*100);
end

% ======== Final Plot =======
figure;
plot(split_ratios*100, EER_results, '-o', 'LineWidth', 2);
hold on;
plot(split_ratios*100, FAR_results, '--s', 'LineWidth', 1.5);
plot(split_ratios*100, FRR_results, ':d', 'LineWidth', 1.5);
hold off;
legend({'EER (%)', 'FAR (%)', 'FRR (%)'}, 'Location', 'northwest');
xlabel('Training Set Ratio (%)');
ylabel('Error Rate (%)');
title('Authentication Performance vs Train/Test Split Ratio');
grid on;
