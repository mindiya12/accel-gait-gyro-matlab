clear; clc;
load('C:/Users/ASUS/Desktop/MATLAB/accel-gait-gyro-matlab/results/extracted_features.mat'); % Update path as needed
X = featureMatrix; Y = allLabels;
K = 5; % or 10 for 10-fold CV
N = size(X,1); 
foldAcc = zeros(K,1);

% --- Custom fold assignment (random, nearly equal size) ---
fold_sizes = floor(N/K)*ones(1,K); fold_sizes(1:mod(N,K)) = fold_sizes(1:mod(N,K)) + 1;
indices = zeros(N,1);
rp = randperm(N); % Shuffle
count = 1;
for k = 1:K
    fold_len = fold_sizes(k);
    indices(rp(count:count+fold_len-1)) = k;
    count = count + fold_len;
end

for fold = 1:K
    test_idx = (indices == fold); train_idx = ~test_idx;
    X_train = X(train_idx,:); Y_train = Y(train_idx);
    X_test = X(test_idx,:);  Y_test = Y(test_idx);
    mu = mean(X_train,1); sigma = std(X_train,0,1); sigma(sigma<1e-8)=1;
    X_train_norm = (X_train-mu)./sigma; X_test_norm = (X_test-mu)./sigma;
    X_train_t = X_train_norm'; X_test_t = X_test_norm';
    y_train_onehot = full(ind2vec(Y_train'));
    net = patternnet([50,30],'trainscg');
    net.trainParam.epochs = 500; net.trainParam.showWindow = false;
    [net,~] = train(net, X_train_t, y_train_onehot);
    output = net(X_test_t); folded_pred = vec2ind(output);
    foldAcc(fold) = mean(folded_pred' == Y_test)*100;
    fprintf('Fold %d Accuracy: %.2f%%\n', fold, foldAcc(fold));
end
meanAcc = mean(foldAcc); stdAcc = std(foldAcc);
fprintf('\nK-Fold CV Average Accuracy: %.2f%% (Std: %.2f%%)\n', meanAcc, stdAcc);
figure;
bar(foldAcc);
xlabel('Fold'); ylabel('Accuracy (%)'); title(sprintf('%d-Fold CV Accuracy',K)); grid on;
