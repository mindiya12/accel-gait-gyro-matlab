% =========================================================================
% NEURAL NETWORK TRAINING - OPTION B (User-Dependent)
% File: train_neural_networks_peruser.m
% =========================================================================
clear; clc;

fprintf('==========================================================\n');
fprintf('NEURAL NETWORK TRAINING FOR USER AUTHENTICATION (PER-USER EVALUATION)\n');
fprintf('==========================================================\n');

% Load preprocessed data (CHANGE PATH IF NEEDED)
fprintf('\n[STEP 1] Loading preprocessed data...\n');
load('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\second_approch\accel-gait-gyro-matlab_2\results\normalized_splits_FINAL.mat');
fprintf('  Option B loaded: %d train, %d test samples\n', ...
    size(X_train_B_norm, 1), size(X_test_B_norm, 1));

fprintf('\n[STEP 2] Configuring Neural Network...\n');
X_train_B_t = X_train_B_norm';  
X_test_B_t = X_test_B_norm';    

y_train_B_onehot = full(ind2vec(y_train_B'));  
y_test_B_onehot = full(ind2vec(y_test_B'));    
y_test_B_class = y_test_B; % actual class labels

hiddenLayerSize = [50 30];
trainFcn = 'trainscg';
net_B = patternnet(hiddenLayerSize, trainFcn);
net_B.trainParam.epochs = 500;
net_B.trainParam.goal = 1e-5;
net_B.trainParam.showWindow = false;
net_B.trainParam.showCommandLine = false;
net_B.divideParam.trainRatio = 0.70;
net_B.divideParam.valRatio = 0.15;
net_B.divideParam.testRatio = 0.15;

fprintf('  Architecture: %d inputs -> [50, 30] hidden -> %d outputs\n', ...
    size(X_train_B_t, 1), size(y_train_B_onehot, 1));
fprintf('  Training function: %s\n', trainFcn);

fprintf('\n[STEP 3] Training Option B Neural Network...\n');
fprintf('  (This may take 1-3 minutes depending on your system)\n\n');
[net_B, tr_B] = train(net_B, X_train_B_t, y_train_B_onehot);
fprintf('  Training complete!\n');
fprintf('  Best performance: %.6f\n', tr_B.best_perf);

fprintf('\n[STEP 4] Evaluating (Per-User EER/FAR/FRR)...\n');
y_pred_B = net_B(X_test_B_t);         % [10 x #test]
y_pred_B_class = vec2ind(y_pred_B);   % predicted label numbers

% --- Per-user EER, FAR, FRR calculation ---
numUsers = 10;
thresholds = 0:0.01:1;

userEERs = zeros(numUsers,1);
userFARs = zeros(numUsers,1);
userFRRs = zeros(numUsers,1);
eerThresholds = zeros(numUsers,1);

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\second_approch\accel-gait-gyro-matlab_2\results\figures\OptionB_peruser';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

for u = 1:numUsers
    % Genuine: test samples from user u
    genuine_idx = (y_test_B_class == u);
    genuine_scores = y_pred_B(u, genuine_idx);

    % Impostor: test samples NOT from user u
    impostor_idx = ~genuine_idx;
    impostor_scores = y_pred_B(u, impostor_idx);

    ALL_FAR = zeros(size(thresholds));
    ALL_FRR = zeros(size(thresholds));

    for i = 1:length(thresholds)
        t = thresholds(i);
        ALL_FAR(i) = sum(impostor_scores >= t) / length(impostor_scores) * 100;
        ALL_FRR(i) = sum(genuine_scores < t) / length(genuine_scores) * 100;
    end

    [~, eer_idx] = min(abs(ALL_FAR - ALL_FRR));
    userEERs(u) = (ALL_FAR(eer_idx) + ALL_FRR(eer_idx)) / 2;
    userFARs(u) = ALL_FAR(eer_idx);
    userFRRs(u) = ALL_FRR(eer_idx);
    eerThresholds(u) = thresholds(eer_idx);

    fprintf('User %2d: EER: %.2f%%  |  FAR: %.2f%%  |  FRR: %.2f%%  |  EER_Thres: %.2f\n', ...
        u, userEERs(u), userFARs(u), userFRRs(u), thresholds(eer_idx));

    % --- Plot and save FAR/FRR curves for this user ---
    fig = figure('Visible','off');
    plot(thresholds, ALL_FAR, '-r','LineWidth',2); hold on;
    plot(thresholds, ALL_FRR, '-b','LineWidth',2);
    plot(thresholds(eer_idx), userEERs(u), 'ko', 'MarkerSize',8, 'MarkerFaceColor','k');
    xlabel('Threshold'); ylabel('Error Rate (%)');
    title(sprintf('User %d: FAR and FRR vs. Threshold with EER', u));
    legend('FAR (False Acceptance Rate)', 'FRR (False Rejection Rate)', 'EER Point');
    grid on;
    saveas(fig, fullfile(resultsFolder, sprintf('User%d_FAR_FRR_EER.png', u)));
    close(fig);
end

fprintf('\n--- MEAN/STANDARD DEVIATION ACROSS USERS ---\n');
fprintf('Mean EER: %.2f%% ± %.2f%%\n', mean(userEERs), std(userEERs));
fprintf('Mean FAR: %.2f%% ± %.2f%%\n', mean(userFARs), std(userFARs));
fprintf('Mean FRR: %.2f%% ± %.2f%%\n', mean(userFRRs), std(userFRRs));

% Save numeric EER/FAR/FRR per user table as CSV
T = table((1:numUsers)', userEERs, userFARs, userFRRs, eerThresholds, ...
    'VariableNames', {'User', 'EER', 'FAR', 'FRR', 'EER_Threshold'});
writetable(T, fullfile(resultsFolder, 'OptionB_peruser_results.csv'));

% -- Optional: Show summary plot (EER per user) --
fig = figure('Visible','off');
bar(userEERs, 'FaceColor', [0.2 0.6 0.8]);
ylabel('EER (%)');
xlabel('User');
title('Equal Error Rate (EER) for Each User');
saveas(fig, fullfile(resultsFolder, 'AllUsers_EER_bar.png'));
close(fig);

% -- Report POOLED system-level EER for context --
all_genuine = []; all_impostor = [];
for u = 1:numUsers
    idx_g = (y_test_B_class == u);
    idx_i = ~idx_g;
    all_genuine = [all_genuine, y_pred_B(u, idx_g)];
    all_impostor = [all_impostor, y_pred_B(u, idx_i)];
end
FAR_pooled = zeros(size(thresholds)); FRR_pooled = zeros(size(thresholds));
for i = 1:length(thresholds)
    t = thresholds(i);
    FAR_pooled(i) = sum(all_impostor >= t) / length(all_impostor) * 100;
    FRR_pooled(i) = sum(all_genuine < t) / length(all_genuine) * 100;
end
[~, eidx_pool] = min(abs(FAR_pooled - FRR_pooled));
EER_pooled = (FAR_pooled(eidx_pool) + FRR_pooled(eidx_pool)) / 2;
fprintf('System-level EER: %.2f%% (FAR: %.2f%% | FRR: %.2f%%) at threshold %.2f\n', ...
    EER_pooled, FAR_pooled(eidx_pool), FRR_pooled(eidx_pool), thresholds(eidx_pool));
fig = figure('Visible','off');
plot(thresholds, FAR_pooled, '-r','LineWidth',2); hold on;
plot(thresholds, FRR_pooled, '-b','LineWidth',2);
plot(thresholds(eidx_pool), EER_pooled, 'ko', 'MarkerSize',8, 'MarkerFaceColor','k');
xlabel('Threshold'); ylabel('Error Rate (%)');
title('System-level FAR/FRR vs. Threshold with EER');
legend('FAR', 'FRR', 'EER Point');
grid on;
saveas(fig, fullfile(resultsFolder, 'Pooled_FAR_FRR_EER.png'));
close(fig);

% -- Save the trained network as before --
fprintf('\n[STEP 5] Saving trained model for Option B...\n');
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\second_approch\accel-gait-gyro-matlab_2\results\trained_model_OptionB.mat', ...
    'net_B', 'tr_B', 'hiddenLayerSize', 'trainFcn');
fprintf('  Saved: trained_model_OptionB.mat\n');

fprintf('\n==========================================================\n');
fprintf('PER-USER EVALUATION COMPLETE. Results and plots saved!\n');
fprintf('==========================================================\n');

