clear; clc;
load('C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/extracted_features.mat');
X = featureMatrix; Y = allLabels;
train_ratio = 0.8;
N = size(X,1); rng(42);
order = randperm(N); n_train = round(train_ratio*N);
train_idx = order(1:n_train); test_idx = order(n_train+1:end);
X_train = X(train_idx,:); Y_train = Y(train_idx);

X_test = X(test_idx,:); Y_test = Y(test_idx);

mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma<1e-8)=1;
X_train_norm = (X_train-mu)./sigma;
X_test_norm = (X_test-mu)./sigma;

feature_sets = {
    1:102,          % Time-domain only
    103:126,        % Frequency-domain only
    127:133,        % Correlation/magnitude
    1:size(X,2)     % All features (baseline)
};
names = {'Time-domain','Frequency-domain','Correlation/mag','All-features'};
numSets = numel(feature_sets);
numUsers = numel(unique(Y));

mean_EERs = zeros(numSets,1);
std_EERs = zeros(numSets,1);
mean_FARs = zeros(numSets,1);
mean_FRRs = zeros(numSets,1);

thresholds = 0:0.01:1;
for i = 1:numSets
    fs = feature_sets{i};
    net = patternnet([50 30],'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    Xtr = X_train_norm(:,fs)'; Xte = X_test_norm(:,fs)';
    y1hot = full(ind2vec(Y_train'));
    [net,~] = train(net, Xtr, y1hot);
    outputs_test = net(Xte);
    y_test_class = Y_test(:)'; % make row for indexing

    % Per-user genuine/impostor EER, FAR, FRR
    userEERs = zeros(numUsers,1);
    userFARs = zeros(numUsers,1);
    userFRRs = zeros(numUsers,1);
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
    mean_EERs(i) = mean(userEERs);
    std_EERs(i) = std(userEERs);
    mean_FARs(i) = mean(userFARs);
    mean_FRRs(i) = mean(userFRRs);

    fprintf('%s Features || Mean EER: %.2f%% ± %.2f%% | FAR: %.2f%% | FRR: %.2f%%\n', ...
        names{i}, mean_EERs(i), std_EERs(i), mean_FARs(i), mean_FRRs(i));
end

figfolder = 'C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/figures/feature_selection_exp';
if ~exist(figfolder, 'dir'), mkdir(figfolder); end

% Essential bar plot (for report)
figure('Visible','off','Position',[100,100,700,400]);
bar(mean_EERs,'FaceColor',[0.2 0.6 0.8]);
hold on;
errorbar(1:numSets, mean_EERs, std_EERs, 'k.', 'LineWidth', 2, 'CapSize', 16);
set(gca,'XTickLabel',names, 'XTick',1:numSets, 'FontSize',12, 'FontWeight','bold');
xlabel('Feature Set','FontSize',13,'FontWeight','bold');
ylabel('Mean EER (%)','FontWeight','bold','FontSize',13);
title('Equal Error Rate (EER) for Different Feature Sets','FontSize',13,'FontWeight','bold');
grid on;
saveas(gcf, fullfile(figfolder,'FeatureSets_PerUserEER.png'));
close;

% Save summary table
T = table(names', mean_EERs, std_EERs, mean_FARs, mean_FRRs, ...
    'VariableNames', {'FeatureSet','Mean_EER','Std_EER','Mean_FAR','Mean_FRR'});
writetable(T, fullfile(figfolder,'FeatureSets_PerUserEER_Summary.csv'));

fprintf('Plot saved: %s\n', fullfile(figfolder,'FeatureSets_PerUserEER.png'));
fprintf('Summary table: %s\n', fullfile(figfolder,'FeatureSets_PerUserEER_Summary.csv'));
