% =========================================================================
% MEMBER 4: EXPERIMENT 4.1 - REGULARIZATION (WEIGHT DECAY) ANALYSIS
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% =========================================================================
clear; clc;
fprintf('===========================================================\n');
fprintf('EXPERIMENT 4.1: WEIGHT DECAY REGULARIZATION\n');
fprintf('Testing different regularization strengths\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n');

% --- Load baseline data ---
load('normalized_splits_FINAL.mat');

% Prepare data
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% --- Test different regularization values ---
reg_values = [0, 0.1, 0.3, 0.5, 0.7];
numUsers = 10;
thresholds = 0:0.01:1;

% Pre-allocate storage arrays
mean_EERs = zeros(1, length(reg_values));
std_EERs = zeros(1, length(reg_values));
mean_FARs = zeros(1, length(reg_values));
std_FARs = zeros(1, length(reg_values));
mean_FRRs = zeros(1, length(reg_values));
std_FRRs = zeros(1, length(reg_values));
accuracies = zeros(1, length(reg_values));

fprintf('\nTesting %d regularization values...\n\n', length(reg_values));

for i = 1:length(reg_values)
    reg = reg_values(i);
    fprintf('--- Testing regularization = %.1f ---\n', reg);
    
    % Create and configure network
    net = patternnet([50, 30], 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % Set regularization parameter
    net.performParam.regularization = reg;
    
    % Disable automatic division to prevent accuracy spam
    net.divideMode = 'none';
    
    % Train the network
    [net, tr] = train(net, X_train, y_train_onehot);
    
    % Evaluate on test set
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    
    % FIXED: Ensure both are row vectors for proper comparison
    y_test_class = y_test(:)';  % Force row vector
    y_pred_class = y_pred_class(:)';  % Force row vector
    
    % --- PER-USER EER/FAR/FRR CALCULATION ---
    userEERs = zeros(numUsers, 1);
    userFARs = zeros(numUsers, 1);
    userFRRs = zeros(numUsers, 1);
    
    for u = 1:numUsers
        % Genuine: test samples from user u (labeled as 1)
        genuine_idx = (y_test_class == u);
        genuine_scores = y_pred(u, genuine_idx);
        
        % Impostor: test samples NOT from user u (labeled as 0)
        impostor_idx = ~genuine_idx;
        impostor_scores = y_pred(u, impostor_idx);
        
        % Calculate FAR and FRR across thresholds
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
        
        % Find EER (where FAR ≈ FRR)
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end
    
    % Store results in arrays
    mean_EERs(i) = mean(userEERs);
    std_EERs(i) = std(userEERs);
    mean_FARs(i) = mean(userFARs);
    std_FARs(i) = std(userFARs);
    mean_FRRs(i) = mean(userFRRs);
    std_FRRs(i) = std(userFRRs);
    
    % FIXED: Accuracy calculation with proper dimension handling
    accuracies(i) = 100 * sum(y_pred_class == y_test_class) / length(y_test_class);
    
    fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(i), std_EERs(i));
    fprintf('  Mean FAR: %.2f%% ± %.2f%%\n', mean_FARs(i), std_FARs(i));
    fprintf('  Mean FRR: %.2f%% ± %.2f%%\n', mean_FRRs(i), std_FRRs(i));
    fprintf('  Accuracy: %.2f%%\n\n', accuracies(i));
end

% --- Create results folder ---
resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\figures\member4_exp1';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% --- Save numerical results ---
results_struct.reg_values = reg_values;
results_struct.mean_EERs = mean_EERs;
results_struct.std_EERs = std_EERs;
results_struct.mean_FARs = mean_FARs;
results_struct.std_FARs = std_FARs;
results_struct.mean_FRRs = mean_FRRs;
results_struct.std_FRRs = std_FRRs;
results_struct.accuracies = accuracies;

save(fullfile(resultsFolder, 'member4_exp1_regularization_results.mat'), 'results_struct');

% --- Generate ONLY essential plot for report: EER vs Regularization ---
figure('Visible', 'off', 'Position', [100, 100, 800, 500]);

errorbar(reg_values, mean_EERs, std_EERs, '-o', 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'b', 'Color', 'b', 'CapSize', 12);
xlabel('Regularization Parameter', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Effect of Weight Decay Regularization on Authentication Security (EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

% Add data labels
for i = 1:length(reg_values)
    text(reg_values(i), mean_EERs(i) + std_EERs(i) + 0.3, ...
        sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, fullfile(resultsFolder, 'Regularization_EER_Analysis.png'));
close;

% --- Generate summary table as CSV for report ---
summary_table = table(reg_values(:), mean_EERs(:), std_EERs(:), ...
    mean_FARs(:), std_FARs(:), ...
    mean_FRRs(:), std_FRRs(:), ...
    accuracies(:), ...
    'VariableNames', {'Regularization', 'Mean_EER', 'Std_EER', ...
    'Mean_FAR', 'Std_FAR', 'Mean_FRR', 'Std_FRR', 'Accuracy'});
writetable(summary_table, fullfile(resultsFolder, 'Regularization_Summary.csv'));

fprintf('\n===========================================================\n');
fprintf('EXPERIMENT 4.1 COMPLETE!\n');
fprintf('===========================================================\n');
fprintf('Per-User Evaluation Results:\n');
[minEER, minIdx] = min(mean_EERs);
fprintf('  Best regularization: %.1f (EER: %.2f%% ± %.2f%%)\n', ...
    reg_values(minIdx), minEER, std_EERs(minIdx));
fprintf('\nResults saved to: %s\n', resultsFolder);
fprintf('  - Numerical results: member4_exp1_regularization_results.mat\n');
fprintf('  - EER plot: Regularization_EER_Analysis.png\n');
fprintf('  - Summary table: Regularization_Summary.csv\n');
fprintf('===========================================================\n\n');
