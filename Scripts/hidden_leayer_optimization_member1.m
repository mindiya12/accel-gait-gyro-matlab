% =========================================================================
% MEMBER 1: EXPERIMENT 1.1 - HIDDEN LAYER ARCHITECTURE TESTING
% Testing 7 different neural network architectures
% =========================================================================
clear; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 1: EXPERIMENT 1.1 - ARCHITECTURE OPTIMIZATION\n');
fprintf('Testing different hidden layer configurations\n');
fprintf('===========================================================\n\n');

% --- Load your preprocessed data ---
fprintf('[STEP 1] Loading preprocessed data...\n');
load('C:\Users\E-TECH\OneDrive - NSBM\Desktop\ai ml\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat');  % Update path as needed

% Prepare data for neural network (transpose: features × samples)
X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;

% Convert labels to one-hot encoding
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
results_arch = struct();

fprintf('[STEP 2] Testing %d architectures...\n\n', num_configs);

% --- Progress bar ---
h = waitbar(0, 'Testing architectures...');

% --- Loop through each architecture ---
for i = 1:num_configs
    config = configs{i};
    
    fprintf('--- Architecture %d/%d: %s ---\n', i, num_configs, config_names{i});
    fprintf('    Layers: %s\n', mat2str(config));
    
    % Update waitbar
    waitbar(i/num_configs, h, sprintf('Testing %s...', config_names{i}));
    
    % Create pattern recognition network
    net = patternnet(config, 'trainscg');  % Scaled conjugate gradient
    
    % Configure training parameters
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;  % Don't show training GUI
    net.trainParam.showCommandLine = false;
    
    % Data division (70% train, 15% validation, 15% internal test)
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Train the network
    tic;  % Start timer
    [net, tr] = train(net, X_train, y_train_onehot);
    train_time = toc;  % End timer
    
    % --- Evaluate on test set ---
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    
    % Calculate accuracy
    accuracy = 100 * mean(y_pred_class == y_test_class);
    
    % Calculate confusion matrix
    C = confusionmat(y_test_class, y_pred_class);
    
    % Calculate per-class metrics
    per_class_acc = zeros(1, 10);
    for u = 1:10
        user_idx = (y_test_class == u);
        if sum(user_idx) > 0
            per_class_acc(u) = 100 * sum(y_pred_class(user_idx) == u) / sum(user_idx);
        end
    end
    
    % Calculate authentication metrics (FAR, FRR, EER estimation)
    max_conf = max(y_pred, [], 1);  % Max confidence per prediction
    
    % Simple EER estimation using threshold sweep
    thresholds = 0:0.01:1;
    FAR = zeros(size(thresholds));
    FRR = zeros(size(thresholds));
    
    for t = 1:length(thresholds)
        thresh = thresholds(t);
        
        % Predictions accepted at this threshold
        accepted = max_conf >= thresh;
        correct_accepted = (y_pred_class == y_test_class) & accepted;
        incorrect_accepted = (y_pred_class ~= y_test_class) & accepted;
        correct_rejected = (y_pred_class == y_test_class) & ~accepted;
        
        % FAR: impostor accepted / total impostor attempts
        total_impostor = sum(y_pred_class ~= y_test_class);
        if total_impostor > 0
            FAR(t) = sum(incorrect_accepted) / total_impostor;
        end
        
        % FRR: genuine rejected / total genuine attempts
        total_genuine = sum(y_pred_class == y_test_class);
        if total_genuine > 0
            FRR(t) = sum(correct_rejected) / total_genuine;
        end
    end
    
    % Find EER (where FAR ≈ FRR)
    [~, eer_idx] = min(abs(FAR - FRR));
    EER = 100 * FAR(eer_idx);
    EER_threshold = thresholds(eer_idx);
    
    % --- Store results ---
    results_arch(i).config = config;
    results_arch(i).name = config_names{i};
    results_arch(i).total_neurons = sum(config);
    results_arch(i).num_layers = length(config);
    results_arch(i).accuracy = accuracy;
    results_arch(i).train_time = train_time;
    results_arch(i).final_perf = tr.best_perf;
    results_arch(i).best_epoch = tr.best_epoch;
    results_arch(i).confusion_matrix = C;
    results_arch(i).per_class_acc = per_class_acc;
    results_arch(i).mean_per_class_acc = mean(per_class_acc);
    results_arch(i).EER = EER;
    results_arch(i).EER_threshold = EER_threshold;
    results_arch(i).mean_confidence = mean(max_conf);
    
    % Display results
    fprintf('    Accuracy: %.2f%%\n', accuracy);
    fprintf('    EER: %.2f%% (threshold: %.2f)\n', EER, EER_threshold);
    fprintf('    Training time: %.2f seconds\n', train_time);
    fprintf('    Best epoch: %d / %d\n', tr.best_epoch, tr.num_epochs);
    fprintf('    Mean confidence: %.4f\n\n', mean(max_conf));
end

close(h);  % Close waitbar

% --- Save all results ---
fprintf('[STEP 3] Saving results...\n');
save('member1_exp1_architecture_results.mat', 'results_arch', 'configs', 'config_names');
fprintf('  ✅ Saved: member1_exp1_architecture_results.mat\n\n');

% --- Create comparison table ---
fprintf('[STEP 4] Creating comparison table...\n');
fprintf('%-25s | Neurons | Layers | Accuracy | EER     | Train Time\n', 'Architecture');
fprintf('----------|---------|--------|----------|---------|------------\n');
for i = 1:num_configs
    fprintf('%-25s | %7d | %6d | %7.2f%% | %6.2f%% | %8.2fs\n', ...
        results_arch(i).name, ...
        results_arch(i).total_neurons, ...
        results_arch(i).num_layers, ...
        results_arch(i).accuracy, ...
        results_arch(i).EER, ...
        results_arch(i).train_time);
end
fprintf('\n');

% --- Identify best architectures ---
[best_acc, idx_acc] = max([results_arch.accuracy]);
[best_eer, idx_eer] = min([results_arch.EER]);
[best_time, idx_time] = min([results_arch.train_time]);

fprintf('BEST CONFIGURATIONS:\n');
fprintf('  Best Accuracy: %s (%.2f%%)\n', results_arch(idx_acc).name, best_acc);
fprintf('  Best EER: %s (%.2f%%)\n', results_arch(idx_eer).name, best_eer);
fprintf('  Fastest Training: %s (%.2fs)\n\n', results_arch(idx_time).name, best_time);

% --- VISUALIZATION 1: Bar chart comparison ---
fprintf('[STEP 5] Creating visualizations...\n');

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Accuracy comparison
subplot(3,2,1);
accuracies = [results_arch.accuracy];
bar(accuracies);
set(gca, 'XTickLabel', strrep(config_names, '-', ' '), 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Test Accuracy by Architecture');
ylim([min(accuracies)-2, 100]);
grid on;

% Subplot 2: EER comparison
subplot(3,2,2);
eers = [results_arch.EER];
bar(eers);
set(gca, 'XTickLabel', strrep(config_names, '-', ' '), 'XTickLabelRotation', 45);
ylabel('EER (%)');
title('Equal Error Rate by Architecture (Lower is Better)');
grid on;

% Subplot 3: Training time comparison
subplot(3,2,3);
times = [results_arch.train_time];
bar(times);
set(gca, 'XTickLabel', strrep(config_names, '-', ' '), 'XTickLabelRotation', 45);
ylabel('Time (seconds)');
title('Training Time by Architecture');
grid on;

% Subplot 4: Neurons vs Accuracy
subplot(3,2,4);
neurons = [results_arch.total_neurons];
plot(neurons, accuracies, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Total Number of Neurons');
ylabel('Accuracy (%)');
title('Network Size vs Performance');
grid on;

% Subplot 5: Layers vs EER
subplot(3,2,5);
layers = [results_arch.num_layers];
plot(layers, eers, '-s', 'LineWidth', 2, 'MarkerSize', 10, 'Color', 'r');
xlabel('Number of Hidden Layers');
ylabel('EER (%)');
title('Network Depth vs Error Rate');
xticks([1 2 3]);
grid on;

% Subplot 6: Performance vs Complexity tradeoff
subplot(3,2,6);
scatter(neurons, accuracies, 100, times, 'filled');
xlabel('Total Neurons');
ylabel('Accuracy (%)');
title('Performance vs Complexity (color = training time)');
colorbar;
colormap('jet');
grid on;

sgtitle('EXPERIMENT 1.1: Architecture Comparison - Member 1', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'member1_exp1_architecture_comparison.png');

fprintf('  ✅ Saved: member1_exp1_architecture_comparison.png\n\n');

fprintf('===========================================================\n');
fprintf('✅ EXPERIMENT 1.1 COMPLETE\n');
fprintf('===========================================================\n');
fprintf('Best architecture for next experiments: %s\n', results_arch(idx_eer).name);
fprintf('This will be used for Experiments 1.2 and 1.3\n');
fprintf('===========================================================\n');
