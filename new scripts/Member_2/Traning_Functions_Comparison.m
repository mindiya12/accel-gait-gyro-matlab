
% FINAL NN TRAINING ARTIFACT (PER-USER EVALUATION ONLY)
clear; clc;

% 1. Load features and labels (replace path if needed)
data = load('C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/extracted_features.mat');
X = data.featureMatrix; % N x D
Y = data.allLabels;     % N x 1  (Users 1-10)

% 2. Random 80/20 train/test split (all users present)
N = size(X,1);
rng(42); order = randperm(N);
n_train = round(0.8*N);
train_idx = order(1:n_train); test_idx = order(n_train+1:end);
X_train = X(train_idx,:); Y_train = Y(train_idx);
X_test  = X(test_idx,:);  Y_test  = Y(test_idx);

% 3. Normalize features with train stats
mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma<1e-8) = 1;
X_train_norm = (X_train-mu)./sigma;
X_test_norm  = (X_test-mu)./sigma;

% 4. Prepare for NN training
X_train_t = X_train_norm'; X_test_t = X_test_norm';
y_train_onehot = full(ind2vec(Y_train'));

% --- Architectures to try ---
arch_list = {[100 50], [50 30], [64 32]};
arch_labels = {'[100 50]', '[50 30]', '[64 32]'};

% Pre-allocate
numUsers = numel(unique(Y)); % usually 10
numArch = numel(arch_list);
mean_EERs = zeros(1,numArch);
std_EERs = zeros(1,numArch);

All_userEER = zeros(numArch, numUsers);

for k = 1:numArch
    fprintf('== Architecture %s ==\n', arch_labels{k});
    net = patternnet(arch_list{k},'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    [net, tr] = train(net, X_train_t, y_train_onehot);
    
    % Predict
    outputs = net(X_test_t);
    pred = vec2ind(outputs);
    
    % Per-user EER/FAR/FRR evaluation
    userEERs = zeros(numUsers,1);
    userFARs = zeros(numUsers,1);
    userFRRs = zeros(numUsers,1);
    thresholds = 0:0.01:1;
    y_test_class = Y_test(:)';  % always row
    
    for u = 1:numUsers
        % User u genuine and impostor calculation
        genuine_idx  = (y_test_class == u);
        impostor_idx = ~genuine_idx;
        genuine_scores  = outputs(u, genuine_idx);
        impostor_scores = outputs(u, impostor_idx);
        FAR = zeros(size(thresholds));
        FRR = zeros(size(thresholds));
        for t_idx = 1:length(thresholds)
            t = thresholds(t_idx);
            FAR(t_idx) = sum(impostor_scores >= t) / length(impostor_scores) * 100;
            FRR(t_idx) = sum(genuine_scores < t) / length(genuine_scores) * 100;
        end
        [~, eer_idx] = min(abs(FAR - FRR));
        userEERs(u) = (FAR(eer_idx) + FRR(eer_idx)) / 2;
        userFARs(u) = FAR(eer_idx);
        userFRRs(u) = FRR(eer_idx);
    end
    
    mean_EERs(k) = mean(userEERs);
    std_EERs(k)  = std(userEERs);
    All_userEER(k, :) = userEERs;
    fprintf('  Mean EER: %.2f%% ± %.2f%%\n', mean_EERs(k), std_EERs(k));
    % For full report, you can print userEERs here if you wish
end

% === PLOT: Mean EER (with std) for each architecture ===
figfolder = 'C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/figures/final_model';
if ~exist(figfolder, 'dir'), mkdir(figfolder); end

figure('Visible','off','Position',[100,100,700,400]);
bar(mean_EERs,'FaceColor',[0.2 0.6 0.9]); hold on;
errorbar(1:numArch, mean_EERs, std_EERs, 'k.', 'LineWidth', 2, 'CapSize', 18);
set(gca,'XTickLabel',arch_labels,'FontSize',12,'FontWeight','bold');
xlabel('NN Architecture','FontSize',13,'FontWeight','bold');
ylabel('Mean EER (%)','FontWeight','bold','FontSize',13);
title('Final Model: Equal Error Rate (EER) for Each NN Architecture', 'FontSize',13,'FontWeight','bold');
grid on;
saveas(gcf, fullfile(figfolder,'FinalModel_EER_vs_Arch.png'));
close;

% === SAVE RESULTS TABLES ===
T = table(arch_labels', mean_EERs(:), std_EERs(:), ...
    'VariableNames', {'Architecture','Mean_EER','Std_EER'});
writetable(T, fullfile(figfolder,'FinalModel_EER_Architectures.csv'));

fprintf('\nPer-user EER, FAR, FRR (for reporting/max fairness) available in All_userEER.\n');
fprintf('Plot saved: %s\n', fullfile(figfolder,'FinalModel_EER_vs_Arch.png'));
fprintf('Summary table: %s\n', fullfile(figfolder,'FinalModel_EER_Architectures.csv'));
