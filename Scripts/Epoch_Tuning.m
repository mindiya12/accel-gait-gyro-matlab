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
names = {'Time-domain','Frequency-domain','Correlation/mag','All'};
accuracies = zeros(numel(feature_sets),1);
for i = 1:numel(feature_sets)
    fs = feature_sets{i};
    net = patternnet([50 30],'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    Xtr = X_train_norm(:,fs)'; Xte = X_test_norm(:,fs)';
    y1hot = full(ind2vec(Y_train'));
    [net,~] = train(net, Xtr, y1hot);
    pred = vec2ind(net(Xte));
    acc = mean(pred' == Y_test)*100;
    accuracies(i) = acc;
    fprintf('%s Features || Accuracy: %.2f%%\n', names{i}, acc);
end
figure;
bar(accuracies);
set(gca,'XTickLabel',names, 'XTick',1:numel(names));
xlabel('Feature Set'); ylabel('Test Accuracy (%)'); grid on;
title('NN Accuracy for Feature Selection Sets');
