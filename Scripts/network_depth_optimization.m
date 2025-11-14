% =========================================================================
% MEMBER 1: EXPERIMENT 1.3 - NETWORK DEPTH ANALYSIS
% Comparing 1-layer, 2-layer, and 3-layer networks with similar neuron count
% =========================================================================
clear all; close all; clc;

fprintf('===========================================================\n');
fprintf('MEMBER 1: EXPERIMENT 1.3 - NETWORK DEPTH ANALYSIS\n');
fprintf('Comparing networks with similar total neurons but varying depth\n');
fprintf('===========================================================\n\n');

% Load preprocessed data
load('normalized_splits_FINAL.mat');   % update path if needed

X_train = X_train_B_norm';
X_test = X_test_B_norm';
y_train = y_train_B;
y_test = y_test_B;
y_train_onehot = full(ind2vec(y_train'));
y_test_onehot = full(ind2vec(y_test'));

% Use best activation from Exp 1.2 (tansig)
activationFcn = 'tansig';

% Define depth configurations (keep total neurons similar—choose baseline)
depth_configs = {
    [80],          
    [50,30],       
    [30,30,20]     
};
depth_names = {'1-Layer (80)', '2-Layer (50/30)', '3-Layer (30/30/20)'};
num_depths = length(depth_configs);
results_depth = struct();

fprintf('[STEP] Testing %d depth configurations...\n\n', num_depths);
for i = 1:num_depths
    config = depth_configs{i};
    fprintf('----------------------------------------------\n');
    fprintf('Depth %d/%d: %s\n', i, num_depths, depth_names{i});
    fprintf('Layers: %s\n', mat2str(config));
    % Create network
    net = patternnet(config, 'trainscg');
    for layer_idx = 1:length(config)
        net.layers{layer_idx}.transferFcn = activationFcn;
    end
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    % Train
    tic;
    [net, tr] = train(net, X_train, y_train_onehot);
    train_time = toc;
    % Evaluate
    y_pred = net(X_test);
    y_pred_class = vec2ind(y_pred);
    y_test_class = vec2ind(y_test_onehot);
    accuracy = 100 * mean(y_pred_class == y_test_class);
    % EER Calculation
    max_conf = max(y_pred, [], 1);
    thresholds = 0:0.01:1;
    FAR = zeros(1, length(thresholds));
    FRR = zeros(1, length(thresholds));
    for t = 1:length(thresholds)
        thresh = thresholds(t);
        accepted = max_conf >= thresh;
        correct = y_pred_class == y_test_class;
        total_impostor = sum(~correct);
        total_genuine = sum(correct);
        if total_impostor > 0
            FAR(t) = sum(~correct & accepted) / total_impostor;
        end
        if total_genuine > 0
            FRR(t) = sum(correct & ~accepted) / total_genuine;
        end
    end
    [~, eer_idx] = min(abs(FAR - FRR));
    EER = 100 * FAR(eer_idx);
    EER_threshold = thresholds(eer_idx);
    % Save results
    results_depth(i).name = depth_names{i};
    results_depth(i).config = config;
    results_depth(i).accuracy = accuracy;
    results_depth(i).EER = EER;
    results_depth(i).EER_threshold = EER_threshold;
    results_depth(i).train_time = train_time;
    results_depth(i).best_epoch = tr.best_epoch;
    results_depth(i).final_perf = tr.best_perf;
    fprintf('  Accuracy: %.2f%%\n', accuracy);
    fprintf('  EER: %.2f%% (threshold: %.2f)\n', EER, EER_threshold);
    fprintf('  Training time: %.2f seconds\n\n', train_time);
end

% Save results
save('member1_exp3_depth_results.mat', 'results_depth', 'depth_configs', 'depth_names');

% Table summary
fprintf('\nSummary Table:\n');
fprintf('%-25s | Accuracy | EER | Train Time\n', 'Architecture');
fprintf('-----------------------------------------------\n');
for i = 1:num_depths
    fprintf('%-25s | %7.2f%% | %6.2f%% | %8.2f\n', results_depth(i).name, results_depth(i).accuracy, results_depth(i).EER, results_depth(i).train_time);
end
fprintf('-----------------------------------------------\n');

% Visualizations (short version)
figure('Position', [400, 200, 800, 600]);
accs = [results_depth.accuracy]; eers = [results_depth.EER]; times = [results_depth.train_time];
subplot(2,2,1); bar(accs); set(gca,'xticklabel',depth_names,'xticklabelrotation',25); ylabel('Accuracy (%)'); title('Test Accuracy');
subplot(2,2,2); bar(eers); set(gca,'xticklabel',depth_names,'xticklabelrotation',25); ylabel('EER (%)'); title('Equal Error Rate');
subplot(2,2,3); bar(times); set(gca,'xticklabel',depth_names,'xticklabelrotation',25); ylabel('Train Time (s)'); title('Training Time');
subplot(2,2,4); plot(1:num_depths, eers, '-os'); xlabel('Config'); ylabel('EER (%)'); title('Depth vs Error Rate'); grid on;
sgtitle('Experiment 1.3: Network Depth Analysis');
saveas(gcf, 'member1_exp3_depth_comparison.png');

fprintf('Graphs saved to member1_exp3_depth_comparison.png\n');

