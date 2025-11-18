% =========================================================================
% MEMBER 1: EXPERIMENT 1.1 - HIDDEN LAYER ARCHITECTURE TESTING
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah's guidance
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 1: EXPERIMENT 1.1 - ARCHITECTURE OPTIMIZATION\n');
fprintf('Testing different hidden layer configurations\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n\n');

% --- Load preprocessed data ---
fprintf('[STEP 1] Loading preprocessed data...\n');
load('C:\Users\E-TECH\OneDrive - NSBM\Desktop\ai ml\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat');

X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

fprintf('  Training samples: %d\n', size(X_train, 2));
fprintf('  Testing samples: %d\n', size(X_test, 2));
fprintf('  Features: %d\n', size(X_train, 1));
fprintf('  Users (classes): %d\n\n', length(unique(y_train)));

% --- Define architectures to test ---
configs = {
    [32],          
    [64],          
    [32, 16],       
    [50, 30],       
    [64, 32],       
    [128, 64],      
    [100, 50, 25]   
};

config_names = {
    'Single-32',
    'Single-64',
    'Two-[32,16]',
    'Two-[50,30]-BASELINE',
    'Two-[64,32]',
    'Two-[128,64]',
    'Three-[100,50,25]'
};

num_configs = length(configs);
numUsers = 10;
thresholds = 0:0.01:1;

% Pre-allocate result arrays
mean_EERs = zeros(1, num_configs);
std_EERs = zeros(1, num_configs);
mean_FARs = zeros(1, num_configs);
mean_FRRs = zeros(1, num_configs);
accuracies = zeros(1, num_configs);
train_times = zeros(1, num_configs);
total_neurons = zeros(1, num_configs);
num_layers = zeros(1, num_configs);

fprintf('[STEP 2] Testing %d architectures...\n\n', num_configs);

h = waitbar(0, 'Testing architectures...');

% --- Loop through each architecture ---
for i = 1:num_configs
    config = configs{i};
    
    fprintf('--- Architecture %d/%d: %s ---\n', i, num_configs, config_names{i});
    fprintf('    Layers: %s\n', mat2str(config));
    
    waitbar(i/num_configs, h, sprintf('Testing %s...', config_names{i}));
    
    % Create network
    net = patternnet(config, 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideMode = 'none';  % Disable auto-division
    
    % Train
    tic;
    [net, tr] = train(net, X_train, y_train_onehot);
    train_times(i) = toc;
    
    % Evaluate
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = y_test(:)';
    y_pred_class = y_pred_class(:)';
    
    % --- PER-USER EER/FAR/FRR CALCULATION ---
    userEERs = zeros(numUsers, 1);
    userFARs = zeros(numUsers, 1);
    userFRRs = zeros(numUsers, 1);
    
    for u = 1:numUsers
        genuine_idx = (y_test_class == u);
        genuine_scores = y_pred(u, genuine_idx);
        
        impostor_idx = ~genuine_idx;
        impostor_scores = y_pred(u, impostor_idx);
        
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
    
    % Store results
    mean_EERs(i) = mean(userEERs);
    std_EERs(i) = std(userEERs);
    mean_FARs(i) = mean(userFARs);
    mean_FRRs(i) = mean(userFRRs);
    accuracies(i) = 100 * sum(y_pred_class == y_test_class) / length(y_test_class);
    total_neurons(i) = sum(config);
    num_layers(i) = length(config);
    
    fprintf('    Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(i), std_EERs(i));
    fprintf('    Mean FAR: %.2f%%\n', mean_FARs(i));
    fprintf('    Mean FRR: %.2f%%\n', mean_FRRs(i));
    fprintf('    Accuracy: %.2f%%\n', accuracies(i));
    fprintf('    Training time: %.2f seconds\n\n', train_times(i));
end

close(h);

% --- Create results folder ---
resultsFolder = 'C:\Users\E-TECH\OneDrive - NSBM\Desktop\ai ml\accel-gait-gyro-matlab\results\figures_1\member1_exp1';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% --- Save numerical results ---
results_struct.config_names = config_names;
results_struct.configs = configs;
results_struct.mean_EERs = mean_EERs;
results_struct.std_EERs = std_EERs;
results_struct.mean_FARs = mean_FARs;
results_struct.mean_FRRs = mean_FRRs;
results_struct.accuracies = accuracies;
results_struct.train_times = train_times;
results_struct.total_neurons = total_neurons;
results_struct.num_layers = num_layers;

save(fullfile(resultsFolder, 'member1_exp1_architecture_results.mat'), 'results_struct');

% --- Comparison table ---
fprintf('[STEP 3] Creating comparison table...\n');
fprintf('%-25s | Neurons | Layers | Mean EER | Std EER | Accuracy | Train Time\n', 'Architecture');
fprintf('--------------------------|---------|--------|----------|---------|----------|------------\n');
for i = 1:num_configs
    fprintf('%-25s | %7d | %6d | %7.2f%% | %6.2f%% | %7.2f%% | %8.2fs\n', ...
        config_names{i}, total_neurons(i), num_layers(i), ...
        mean_EERs(i), std_EERs(i), accuracies(i), train_times(i));
end
fprintf('\n');

% --- Identify best architectures ---
[best_eer, idx_eer] = min(mean_EERs);
[best_acc, idx_acc] = max(accuracies);

fprintf('BEST CONFIGURATIONS:\n');
fprintf('  Best EER: %s (%.2f%% ± %.2f%%)\n', config_names{idx_eer}, best_eer, std_EERs(idx_eer));
fprintf('  Best Accuracy: %s (%.2f%%)\n\n', config_names{idx_acc}, best_acc);

% --- ESSENTIAL PLOT 1: EER vs Architecture (MAIN RESULT) ---
fprintf('[STEP 4] Creating essential visualizations for report...\n');

figure('Visible', 'off', 'Position', [100, 100, 1000, 500]);
errorbar(1:num_configs, mean_EERs, std_EERs, '-o', 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'b', 'Color', 'b', 'CapSize', 12);
set(gca, 'XTick', 1:num_configs, 'XTickLabel', strrep(config_names, '-', ' '), ...
    'XTickLabelRotation', 45, 'FontSize', 11);
xlabel('Architecture', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Effect of Neural Network Architecture on Authentication Security (EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Add value labels
for i = 1:num_configs
    text(i, mean_EERs(i) + std_EERs(i) + 0.3, sprintf('%.2f%%', mean_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, fullfile(resultsFolder, 'Architecture_EER_Comparison.png'));
close;

% --- ESSENTIAL PLOT 2: Network Complexity vs EER (INSIGHT) ---
figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
scatter(total_neurons, mean_EERs, 150, num_layers, 'filled');
xlabel('Total Number of Neurons', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Network Complexity vs Authentication Security', 'FontSize', 13, 'FontWeight', 'bold');
colorbar('Ticks', [1 2 3], 'TickLabels', {'1 Layer', '2 Layers', '3 Layers'});
colormap('jet');
grid on;
set(gca, 'FontSize', 11);

% Label best architecture
[~, best_idx] = min(mean_EERs);
text(total_neurons(best_idx), mean_EERs(best_idx) + 0.5, ...
    sprintf('  ← %s\n  (Best: %.2f%%)', config_names{best_idx}, mean_EERs(best_idx)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

saveas(gcf, fullfile(resultsFolder, 'Complexity_vs_EER.png'));
close;

% --- Save summary table as CSV ---
% --- Save summary table as CSV ---
config_names_col = config_names(:);               % ensure 7x1 cell array
total_neurons_col = total_neurons(:);             % ensure 7x1 double
num_layers_col = num_layers(:);
mean_EERs_col = mean_EERs(:);
std_EERs_col = std_EERs(:);
mean_FARs_col = mean_FARs(:);
mean_FRRs_col = mean_FRRs(:);
accuracies_col = accuracies(:);
train_times_col = train_times(:);

summary_table = table(config_names_col, total_neurons_col, num_layers_col, ...
    mean_EERs_col, std_EERs_col, mean_FARs_col, mean_FRRs_col, ...
    accuracies_col, train_times_col, ...
    'VariableNames', {'Architecture', 'Total_Neurons', 'Num_Layers', ...
    'Mean_EER', 'Std_EER', 'Mean_FAR', 'Mean_FRR', 'Accuracy', 'Train_Time'});

writetable(summary_table, fullfile(resultsFolder, 'Architecture_Summary.csv'));


fprintf('  ✅ Saved: Architecture_EER_Comparison.png\n');
fprintf('  ✅ Saved: Complexity_vs_EER.png\n');
fprintf('  ✅ Saved: Architecture_Summary.csv\n\n');

fprintf('===========================================================\n');
fprintf('✅ EXPERIMENT 1.1 COMPLETE\n');
fprintf('===========================================================\n');
fprintf('Best architecture: %s\n', config_names{idx_eer});
fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(idx_eer), std_EERs(idx_eer));
fprintf('  Accuracy: %.2f%%\n', accuracies(idx_eer));
fprintf('\nResults saved to: %s\n', resultsFolder);
fprintf('===========================================================\n');
