% === FINAL NN TRAINING ARTIFACT ===
clear; clc;
% 1. Load features and labels (replace path if needed)
data = load('C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/extracted_features.mat');
X = data.featureMatrix; % N x D
Y = data.allLabels;    % N x 1

% 2. Random 80/20 train/test split (all users present)
N = size(X,1);
rng(42); order = randperm(N);
n_train = round(0.8*N);
train_idx = order(1:n_train); test_idx = order(n_train+1:end);
X_train = X(train_idx,:); Y_train = Y(train_idx);
X_test = X(test_idx,:);  Y_test = Y(test_idx);

% 3. Normalize features with train stats
mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma<1e-8) = 1;
X_train_norm = (X_train-mu)./sigma;
X_test_norm = (X_test-mu)./sigma;

% 4. Prepare for NN training
X_train_t = X_train_norm'; X_test_t = X_test_norm';
y_train_onehot = full(ind2vec(Y_train'));

% 5. Define and train patternnet with 'trainscg' (recommended)
hiddenLayerSizes = [100 50]; % or [50 30], [64 32], [128 64 32]
net = patternnet(hiddenLayerSizes,'trainscg');
net.trainParam.epochs = 500;
net.trainParam.showWindow = false;
[net, tr] = train(net, X_train_t, y_train_onehot);

% 6. Test and report
outputs = net(X_test_t);
pred = vec2ind(outputs);
accuracy = mean(pred' == Y_test) * 100;
fprintf('Test accuracy: %.2f%%\n', accuracy);

% Confusion matrix
figure;
confusionchart(Y_test, pred');
title('Test Set Confusion Matrix');

% Optional: Try a few architectures
for i = 1:3
    sizes = {[50 30], [64 32], [100 50]};
    net = patternnet(sizes{i},'trainscg');
    net.trainParam.epochs = 500; net.trainParam.showWindow = false;
    [net,~] = train(net, X_train_t, y_train_onehot);
    p = vec2ind(net(X_test_t));
    fprintf('Arch %s accuracy: %.2f%%\n', mat2str(sizes{i}), mean(p'==Y_test)*100);
end
