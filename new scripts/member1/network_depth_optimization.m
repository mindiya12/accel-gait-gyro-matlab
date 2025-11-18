% =========================================================================
% MEMBER 1: EXPERIMENT 1.3 - NETWORK DEPTH ANALYSIS
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 1: EXPERIMENT 1.3 - NETWORK DEPTH ANALYSIS\n');
fprintf('Comparing networks with similar total neurons but varying depth\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n\n');

% -------------------------------------------------------------------------
% STEP 1: LOAD DATA
% -------------------------------------------------------------------------
load('normalized_splits_FINAL.mat');   % update path if needed

X_train = X_train_B_norm';
X_test  = X_test_B_norm';
y_train = y_train_B;
y_test  = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot  = full(ind2vec(y_test'));

fprintf('  Training samples: %d\n', size(X_train, 2));
fprintf('  Testing samples : %d\n', size(X_test, 2));
fprintf('  Users (classes) : %d\n\n', numel(unique(y_train)));

% -------------------------------------------------------------------------
% STEP 2: CONFIGURATIONS (DEPTH)
% -------------------------------------------------------------------------

activationFcn = 'tansig';   % best from Exp 1.2

depth_configs = {
    [80]          % 1 layer
    [50, 30]      % 2 layers (baseline)
    [30, 30, 20]  % 3 layers
};
depth_names = {'1-Layer (80)', '2-Layer (50/30)', '3-Layer (30/30/20)'};
num_depths  = numel(depth_configs);

numUsers   = 10;
thresholds = 0:0.01:1;

% Preallocate metrics
mean_EERs = zeros(1, num_depths);
mean_FARs = zeros(1, num_depths);
mean_FRRs = zeros(1, num_depths);
accuracies = zeros(1, num_depths);
train_times = zeros(1, num_depths);

fprintf('[STEP 2] Testing %d depth configurations...\n\n', num_depths);

for i = 1:num_depths
    config = depth_configs{i};
    fprintf('----------------------------------------------\n');
    fprintf('Depth %d/%d: %s\n', i, num_depths, depth_names{i});
    fprintf('Layers: %s\n', mat2str(config));
    
    % ---------------------------------------------------------------------
    % CREATE + TRAIN NETWORK
    % ---------------------------------------------------------------------
    net = patternnet(config, 'trainscg');
    for l = 1:numel(config)
        net.layers{l}.transferFcn = activationFcn;
    end
    net.trainParam.epochs          = 500;
    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;
    net.divideMode                 = 'none';     % no internal split
    
    tic;
    [net, tr] = train(net, X_train, y_train_onehot);
    train_time = toc;
    train_times(i) = train_time;
    
    % ---------------------------------------------------------------------
    % EVALUATION: PER-USER FAR/FRR/EER
    % ---------------------------------------------------------------------
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = y_test(:)';         % row vector
    y_pred_class = y_pred_class(:)';   % row vector
    
    % Accuracy (for context)
    accuracies(i) = 100 * sum(y_pred_class == y_test_class) / numel(y_test_class);
    
    userEERs = zeros(numUsers,1);
    userFARs = zeros(numUsers,1);
    userFRRs = zeros(numUsers,1);
    
    for u = 1:numUsers
        genuine_idx    = (y_test_class == u);   % genuine attempts for user u
        impostor_idx   = ~genuine_idx;          % impostor attempts
        genuine_scores = y_pred(u, genuine_idx);
        impostor_scores = y_pred(u, impostor_idx);
        
        FAR = zeros(size(thresholds));
        FRR = zeros(size(thresholds));
        
        for t_idx = 1:numel(thresholds)
            t = thresholds(t_idx);
            if ~isempty(impostor_scores)
                FAR(t_idx) = sum(impostor_scores >= t) / numel(impostor_scores) * 100;
            end
            if ~isempty(genuine_scores)
                FRR(t_idx) = sum(genuine_scores < t) / numel(genuine_scores) * 100;
            end
        end
        
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end
    
    mean_EERs(i) = mean(userEERs);
    mean_FARs(i) = mean(userFARs);
    mean_FRRs(i) = mean(userFRRs);
    
    fprintf('  Mean EER : %.2f%%\n', mean_EERs(i));
    fprintf('  Mean FAR : %.2f%%\n', mean_FARs(i));
    fprintf('  Mean FRR : %.2f%%\n', mean_FRRs(i));
    fprintf('  Accuracy : %.2f%%\n', accuracies(i));
    fprintf('  Training time: %.2f seconds\n\n', train_times(i));
end

% -------------------------------------------------------------------------
% STEP 3: SAVE NUMERICAL RESULTS
% -------------------------------------------------------------------------
resultsFolder = 'results\figures\member1_exp3';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

results_depth.depth_names  = depth_names;
results_depth.depth_configs = depth_configs;
results_depth.mean_EERs    = mean_EERs;
results_depth.mean_FARs    = mean_FARs;
results_depth.mean_FRRs    = mean_FRRs;
results_depth.accuracies   = accuracies;
results_depth.train_times  = train_times;

save(fullfile(resultsFolder, 'member1_exp3_depth_results.mat'), 'results_depth');

% Text summary
fprintf('\nSummary Table (per-user metrics):\n');
fprintf('--------------------------------------------------------------\n');
fprintf('%-20s | EER     | FAR     | FRR     | Acc     | Time (s)\n', 'Architecture');
fprintf('--------------------------------------------------------------\n');
for i = 1:num_depths
    fprintf('%-20s | %6.2f%% | %6.2f%% | %6.2f%% | %6.2f%% | %7.2f\n', ...
        depth_names{i}, mean_EERs(i), mean_FARs(i), mean_FRRs(i), ...
        accuracies(i), train_times(i));
end
fprintf('--------------------------------------------------------------\n\n');

% CSV table
depth_names_col = depth_names(:);
mean_EERs_col   = mean_EERs(:);
mean_FARs_col   = mean_FARs(:);
mean_FRRs_col   = mean_FRRs(:);
acc_col         = accuracies(:);
time_col        = train_times(:);

summary_table = table(depth_names_col, mean_EERs_col, mean_FARs_col, ...
    mean_FRRs_col, acc_col, time_col, ...
    'VariableNames', {'Architecture', 'Mean_EER', 'Mean_FAR', ...
    'Mean_FRR', 'Accuracy', 'Train_Time'});
writetable(summary_table, fullfile(resultsFolder, 'Depth_Summary.csv'));

% -------------------------------------------------------------------------
% STEP 4: ESSENTIAL FIGURE FOR REPORT (EER vs DEPTH)
% -------------------------------------------------------------------------
fprintf('[STEP] Creating essential EER vs Depth plot...\n');

fig = figure('Visible', 'off', 'Position', [200, 200, 800, 500]);
bar(mean_EERs, 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTick', 1:num_depths, 'XTickLabel', depth_names, ...
    'XTickLabelRotation', 25, 'FontSize', 11);
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Network Depth Configuration', 'FontSize', 12, 'FontWeight', 'bold');
title('Effect of Network Depth on Authentication Security (EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;

for i = 1:num_depths
    text(i, mean_EERs(i) + 0.3, sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(fig, fullfile(resultsFolder, 'Depth_EER_Comparison.png'));
close(fig);

fprintf('Results saved to: %s\n', resultsFolder);
fprintf('  - member1_exp3_depth_results.mat\n');
fprintf('  - Depth_Summary.csv\n');
fprintf('  - Depth_EER_Comparison.png\n');
fprintf('===========================================================\n');
fprintf('✅ EXPERIMENT 1.3 COMPLETE (Per-User Evaluation)\n');
fprintf('===========================================================\n');
