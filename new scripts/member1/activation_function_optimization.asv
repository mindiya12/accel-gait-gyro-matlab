% =========================================================================
% MEMBER 1: EXPERIMENT 1.2 - ACTIVATION FUNCTION TESTING
% Per-User Evaluation (Genuine=1, Impostor=0) as per Dr. Neamah
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 1: EXPERIMENT 1.2 - ACTIVATION FUNCTION OPTIMIZATION\n');
fprintf('Testing activation functions on Two-[50,30] architecture\n');
fprintf('Per-User Evaluation: Genuine(1) vs Impostor(0)\n');
fprintf('===========================================================\n\n');

%% STEP 1: LOAD DATA
fprintf('[STEP 1] Loading preprocessed data...\n');

data_path = 'normalized_splits_FINAL.mat';  % update if needed
if ~exist(data_path, 'file')
    error('File not found: %s', data_path);
end
load(data_path);

X_train = X_train_B_norm';
X_test  = X_test_B_norm';
y_train = y_train_B;
y_test  = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot  = full(ind2vec(y_test'));

fprintf('  Training samples: %d\n', size(X_train, 2));
fprintf('  Testing samples : %d\n\n', size(X_test, 2));

%% STEP 2: FIXED ARCHITECTURE
fprintf('[STEP 2] Using best architecture from Experiment 1.1...\n');
best_config = [50, 30];
fprintf('  Architecture: %s\n\n', mat2str(best_config));

%% STEP 3: ACTIVATION FUNCTIONS
fprintf('[STEP 3] Defining activation functions to test...\n');

activations = {'tansig', 'logsig', 'purelin', 'poslin'};
activation_names = {
    'Tanh (Default)'
    'Sigmoid'
    'Linear'
    'Positive Linear'
};
activation_descriptions = {
    'Hyperbolic tangent ([-1,1])'
    'Logistic sigmoid ([0,1])'
    'Linear (no non-linearity)'
    'ReLU-like ([0,inf))'
};
num_activations = numel(activations);

for i = 1:num_activations
    fprintf('  %d. %s - %s\n', i, activation_names{i}, activation_descriptions{i});
end
fprintf('\n');

%% STEP 4: TRAIN + PER-USER EVALUATION
fprintf('[STEP 4] Training & evaluating each activation (per-user EER/FAR/FRR)...\n\n');

numUsers   = 10;
thresholds = 0:0.01:1;

results_activation = struct([]);

for i = 1:num_activations
    activ = activations{i};
    fprintf('========================================\n');
    fprintf('Activation %d/%d: %s (%s)\n', i, num_activations, activation_names{i}, activ);
    fprintf('========================================\n');

    % Create network
    net = patternnet(best_config, 'trainscg');
    for l = 1:numel(best_config)
        net.layers{l}.transferFcn = activ;
    end
    net.trainParam.epochs           = 500;
    net.trainParam.showWindow       = false;
    net.trainParam.showCommandLine  = false;
    net.divideMode                  = 'none';   % no automatic division

    % Train
    fprintf('  Training...\n');
    tic;
    try
        [net, tr] = train(net, X_train, y_train_onehot);
        train_time = toc;
        success = true;
        fprintf('  Training complete (%.2f s)\n', train_time);
    catch ME
        train_time = toc;
        success = false;
        fprintf('  Training FAILED: %s\n\n', ME.message);
    end

    % Default values
    accuracy    = 0;
    mean_EER    = 100;
    mean_FAR    = 100;
    mean_FRR    = 100;
    best_epoch  = NaN;
    final_perf  = NaN;

    if success
        fprintf('  Evaluating...\n');
        y_pred = net(X_test);
        y_pred_class = vec2ind(y_pred);
        y_test_class = y_test(:)';        % row
        y_pred_class = y_pred_class(:)';  % row

        % accuracy (for context)
        accuracy = 100 * sum(y_pred_class == y_test_class) / numel(y_test_class);

        % per-user EER/FAR/FRR
        userEERs = zeros(numUsers,1);
        userFARs = zeros(numUsers,1);
        userFRRs = zeros(numUsers,1);

        for u = 1:numUsers
            genuine_idx   = (y_test_class == u);
            impostor_idx  = ~genuine_idx;
            genuine_scores  = y_pred(u, genuine_idx);
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

            [~, eer_idx]   = min(abs(FAR - FRR));
            userEERs(u)    = (FAR(eer_idx) + FRR(eer_idx)) / 2;
            userFARs(u)    = FAR(eer_idx);
            userFRRs(u)    = FRR(eer_idx);
        end

        mean_EER = mean(userEERs);
        mean_FAR = mean(userFARs);
        mean_FRR = mean(userFRRs);

        best_epoch = tr.best_epoch;
        final_perf = tr.best_perf;

        fprintf('  Mean EER : %.2f%%\n', mean_EER);
        fprintf('  Mean FAR : %.2f%%\n', mean_FAR);
        fprintf('  Mean FRR : %.2f%%\n', mean_FRR);
        fprintf('  Accuracy : %.2f%%\n', accuracy);
        fprintf('  Best epoch: %d / %d\n\n', tr.best_epoch, tr.num_epochs);
    end

    % store
    results_activation(i).name        = activation_names{i};
    results_activation(i).function    = activ;
    results_activation(i).description = activation_descriptions{i};
    results_activation(i).success     = success;
    results_activation(i).mean_EER    = mean_EER;
    results_activation(i).mean_FAR    = mean_FAR;
    results_activation(i).mean_FRR    = mean_FRR;
    results_activation(i).accuracy    = accuracy;
    results_activation(i).train_time  = train_time;
    results_activation(i).best_epoch  = best_epoch;
    results_activation(i).final_perf  = final_perf;
end

fprintf('All activations tested.\n\n');

%% STEP 5: SAVE RESULTS

resultsFolder = 'results\figures\member1_exp2';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

save(fullfile(resultsFolder, 'member1_exp2_activation_results.mat'), ...
     'results_activation', 'best_config', 'activations', 'activation_names');

%% STEP 6: SUMMARY TABLE (TEXT + CSV)

fprintf('[STEP 6] Summary table (per-user metrics)...\n');
fprintf('====================================================================\n');
fprintf('%-18s | EER     | FAR     | FRR     | Acc     | Time (s)\n', 'Activation');
fprintf('====================================================================\n');

valid_idx = find([results_activation.success]);
for k = valid_idx
    r = results_activation(k);
    fprintf('%-18s | %6.2f%% | %6.2f%% | %6.2f%% | %6.2f%% | %7.2f\n', ...
        r.name, r.mean_EER, r.mean_FAR, r.mean_FRR, r.accuracy, r.train_time);
end
fprintf('====================================================================\n\n');

% CSV summary
act_names_col = {results_activation.name}';
mean_EER_col  = [results_activation.mean_EER]';
mean_FAR_col  = [results_activation.mean_FAR]';
mean_FRR_col  = [results_activation.mean_FRR]';
acc_col       = [results_activation.accuracy]';
time_col      = [results_activation.train_time]';

summary_table = table(act_names_col, mean_EER_col, mean_FAR_col, ...
    mean_FRR_col, acc_col, time_col, ...
    'VariableNames', {'Activation', 'Mean_EER', 'Mean_FAR', ...
    'Mean_FRR', 'Accuracy', 'Train_Time'});
writetable(summary_table, fullfile(resultsFolder, 'Activation_Summary.csv'));

%% STEP 7: ESSENTIAL FIGURE FOR REPORT (EER vs ACTIVATION)

fprintf('[STEP 7] Creating essential EER plot...\n');

fig = figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot_EERs = [results_activation.mean_EER];
bar(plot_EERs, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTick', 1:num_activations, 'XTickLabel', activation_names, ...
    'XTickLabelRotation', 20, 'FontSize', 11);
ylabel('Equal Error Rate - EER (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Activation Function', 'FontSize', 12, 'FontWeight', 'bold');
title('Effect of Activation Function on Authentication Security (EER)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on;

% labels
for i = 1:num_activations
    text(i, plot_EERs(i) + 0.3, sprintf('%.2f%%', plot_EERs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(fig, fullfile(resultsFolder, 'Activation_EER_Comparison.png'));
close(fig);

%% FINAL SUMMARY
fprintf('Results saved to: %s\n', resultsFolder);
fprintf('  - member1_exp2_activation_results.mat\n');
fprintf('  - Activation_Summary.csv\n');
fprintf('  - Activation_EER_Comparison.png\n');
fprintf('===========================================================\n');
fprintf('✅ EXPERIMENT 1.2 COMPLETE (Per-User Evaluation)\n');
fprintf('===========================================================\n');
