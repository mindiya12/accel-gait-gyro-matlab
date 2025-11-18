% --- VARYING TRAIN/TEST SPLITS FOR NN AUTHENTICATION (PER-USER EVALUATION) ---
clear; clc;

% Load the extracted features and labels
data = load('C:\Users\ASUS\Desktop\MATLAB\accel-gait-gyro-matlab\results\extracted_features.mat');
X = data.featureMatrix;          % Features: N × D
Y = data.allLabels;              % Labels: N × 1

split_ratios = [0.60 0.70 0.75 0.80 0.85 0.90];
numUsers = numel(unique(Y)); % e.g., 10 users

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
    y_test_class = Y_test(:)'; % always row for indexing

    % ======== Train Neural Network =======
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.divideMode = 'none';
    net.trainParam.showWindow = false;

    [net, tr] = train(net, X_train_t, y_train_onehot);

    % ======== Test and Evaluate (PER-USER) =======
    outputs_test = net(X_test_t);

    userEERs = zeros(numUsers,1);
    userFARs = zeros(numUsers,1);
    userFRRs = zeros(numUsers,1);
    thresholds = 0:0.01:1;

    for u = 1:numUsers
        genuine_idx = (y_test_class == u);
        impostor_idx = ~genuine_idx;
        genuine_scores = outputs_test(u, genuine_idx);
        impostor_scores = outputs_test(u, impostor_idx);
        FAR = zeros(size(thresholds));
        FRR = zeros(size(thresholds));
        for t_idx = 1:length(thresholds)
            t = thresholds(t_idx);
            if ~isempty(impostor_scores)
                FAR(t_idx) = sum(impostor_scores >= t) / length(impostor_scores) * 100;
            end
            if ~isempty(genuine_scores)
                FRR(t_idx) = sum(genuine_scores < t) / length(genuine_scores) * 100;
            end
        end
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end

    % Store MEAN (across users, only per-user, not pooled!) for this split
    EER_results(idx) = mean(userEERs);
    FAR_results(idx) = mean(userFARs);
    FRR_results(idx) = mean(userFRRs);

    fprintf('Split %.0f%% train: Mean EER = %.2f%% | Mean FAR = %.2f%% | Mean FRR = %.2f%%\n', ...
        ratio*100, mean(userEERs), mean(userFARs), mean(userFRRs));
end

% ======== Final Plot (ESSENTIAL! For report) =======
figfolder = 'C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/figures/split_exp';
if ~exist(figfolder, 'dir'), mkdir(figfolder); end

figure('Visible','off','Position',[100, 100, 700, 400]);
plot(split_ratios*100, EER_results, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0 0.6 0.8]);
hold on;
plot(split_ratios*100, FAR_results, '--s', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.9 0 0]);
plot(split_ratios*100, FRR_results, ':d', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0 0.4 0]);
xticks(split_ratios*100);
legend({'EER (%)', 'FAR (%)', 'FRR (%)'}, 'Location', 'northwest','FontSize',12);
xlabel('Training Set Ratio (%)','FontSize',13,'FontWeight','bold');
ylabel('Error Rate (%)','FontWeight','bold','FontSize',13);
title('Authentication: EER, FAR, and FRR vs Train/Test Split Ratio', 'FontSize',13,'FontWeight','bold');
grid on;
saveas(gcf, fullfile(figfolder,'SplitRatios_EER_FAR_FRR.png'));
close;

% ======= Save summary CSV =======
T = table(split_ratios(:)*100, EER_results(:), FAR_results(:), FRR_results(:), ...
    'VariableNames', {'TrainRatio_pct','Mean_EER','Mean_FAR','Mean_FRR'});
writetable(T, fullfile(figfolder,'SplitRatios_PerUserEER_summary.csv'));

fprintf('\nPlot saved: %s\n', fullfile(figfolder,'SplitRatios_EER_FAR_FRR.png'));
fprintf('Summary table: %s\n', fullfile(figfolder,'SplitRatios_PerUserEER_summary.csv'));
