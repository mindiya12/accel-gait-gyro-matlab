% CROSS-USER AUTHENTICATION AND METRIC EVALUATION ARTIFACT
% (Option A - True Authentication)
% Ref: Ichtiaroglou (2020), AUToSen (2023)
clear; clc;

% === Step 1: Load cross-user split data ===
% Use normalized features and labels for Option A (Users 1-8 train, 9-10 test)
load('C:\Users\ASUS\Desktop\MATLAB\accel-gait-gyro-matlab\results\normalized_splits_FINAL.mat'); % <-- Use your actual filepath
X_train = X_train_A_norm; y_train = y_train_A;
X_test  = X_test_A_norm;  y_test  = y_test_A;

classes = unique(y_train);  % [1 2 ... 8]
X_train_t = X_train'; X_test_t = X_test';
y_train_onehot = full(ind2vec(y_train'));

% === Step 2: Train neural network on enrolled users ===
net = patternnet([50 30], 'trainscg');
net.trainParam.epochs = 500;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.trainParam.showWindow = false;
[net, tr] = train(net, X_train_t, y_train_onehot);

% === Step 3: Evaluate on genuine and impostor samples ===
train_scores = net(X_train_t); % Users 1-8: genuine
maxConf_genuine = max(train_scores, [], 1);
test_scores = net(X_test_t);   % Users 9-10: impostors
maxConf_impostor = max(test_scores, [], 1);

% === Step 4: Compute FAR/FRR/EER vs threshold ===
thresholds = 0:0.01:1;
FAR = zeros(size(thresholds)); FRR = zeros(size(thresholds));
for t = 1:length(thresholds)
    th = thresholds(t);
    FRR(t) = sum(maxConf_genuine < th) / length(maxConf_genuine);
    FAR(t) = sum(maxConf_impostor >= th) / length(maxConf_impostor);
end
[~, eer_idx] = min(abs(FAR - FRR)); EER = mean([FAR(eer_idx), FRR(eer_idx)]); EER_th = thresholds(eer_idx);

fprintf('EER = %.2f%% at threshold = %.2f\n', EER*100, EER_th);

% === Step 5: Plot metric curves ===
figure;
plot(thresholds, FAR*100, 'r-', 'LineWidth',2); hold on;
plot(thresholds, FRR*100, 'b-', 'LineWidth',2);
plot(EER_th, EER*100, 'ko', 'MarkerSize',8, 'MarkerFaceColor','y');
legend('FAR','FRR','EER'); xlabel('Threshold'); ylabel('Error Rate (%)');
title('Cross-User Authentication: FAR/FRR/EER vs. Threshold'); grid on;

% === Step 6: Report accuracy and confusion matrix ===
outputs_test = net(X_test_t);
predict_label = vec2ind(outputs_test);
accuracy = mean(predict_label' == y_test)*100;
fprintf('Closed-set test accuracy: %.2f%%\n', accuracy);
figure;
confusionchart(y_test, predict_label');
title('Test Users Confusion Matrix');
