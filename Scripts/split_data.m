% ========================================================
% DATA SPLITTING (User-wise and Random splits)
% ========================================================
load('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results\extracted_features.mat'); 
X = featureMatrix;
y = allLabels;

% --- Option A: User-wise (users 1-8 train, 9-10 test) ---
trainIdxA = y <= 8;
testIdxA = y > 8;
X_train_A = X(trainIdxA,:);
y_train_A = y(trainIdxA);
X_test_A = X(testIdxA,:);
y_test_A = y(testIdxA);

% --- Option B: within-user 80/20 random split ---
X_train_B = [];
y_train_B = [];
X_test_B = [];
y_test_B = [];
for u = 1:10
    idx = (y == u);
    Xu = X(idx, :);
    yu = y(idx);
    N = size(Xu, 1);
    nTrain = round(0.8*N);
    order = randperm(N);
    X_train_B = [X_train_B; Xu(order(1:nTrain),:)];
    y_train_B = [y_train_B; yu(order(1:nTrain))];
    X_test_B = [X_test_B; Xu(order(nTrain+1:end),:)];
    y_test_B = [y_test_B; yu(order(nTrain+1:end))];
end

save('split_data.mat', 'X_train_A', 'y_train_A', 'X_test_A', 'y_test_A', ...
    'X_train_B', 'y_train_B', 'X_test_B', 'y_test_B');
