% === OPTION B: Within-User NN Optimization (single artifact) ===
clear; clc;
load('C:\Users\ASUS\Desktop\MATLAB\accel-gait-gyro-matlab\results\extracted_features.mat'); % Adjust path if needed
X = featureMatrix;
Y = allLabels;
N = size(X,1); rng(42);
order = randperm(N); train_ratio = 0.8;
n_train = round(N*train_ratio);
train_idx = order(1:n_train); test_idx = order(n_train+1:end);
X_train = X(train_idx,:); Y_train = Y(train_idx);
X_test  = X(test_idx,:);  Y_test  = Y(test_idx);
mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma < 1e-8) = 1;
X_train_norm = (X_train-mu)./sigma;
X_test_norm  = (X_test-mu)./sigma;
architectures = { [50,30], [64,32], [100,50], [128,64,32] };
for i = 1:length(architectures)
    net = patternnet(architectures{i}, 'trainscg');
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    Xtr = X_train_norm'; Xte = X_test_norm';
    ytr1hot = full(ind2vec(Y_train'));
    [net, tr] = train(net, Xtr, ytr1hot);
    output = net(Xte);
    pred = vec2ind(output); acc = mean(pred'==Y_test)*100;
    fprintf('Arch %s - Accuracy: %.2f%%\n', mat2str(architectures{i}), acc);
end
% Final confusion matrix using best model:
best_arch = architectures{2}; % For example, [64 32]
net = patternnet(best_arch, 'trainscg');
net.trainParam.epochs = 500; net.trainParam.showWindow = false;
Xtr = X_train_norm'; Xte = X_test_norm';
ytr1hot = full(ind2vec(Y_train'));
[net, tr] = train(net, Xtr, ytr1hot);
output = net(Xte); pred = vec2ind(output);
figure; confusionchart(Y_test, pred'); title('Option B: Test Confusion Matrix');
