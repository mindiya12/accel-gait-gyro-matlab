% train_baseline_model.m
% Train neural network using Statistics and Machine Learning Toolbox

% Load normalized training data
load('normalized_splits.mat', 'X_train_B_norm', 'y_train_B');

fprintf('Training feedforward neural network...\n');
fprintf('Architecture: Input(133) -> 64 -> 32 -> Output(10)\n');

% Train neural network classifier with fitcnet
net = fitcnet(X_train_B_norm, y_train_B, ...
    'LayerSizes', [64, 32], ...           % Two hidden layers
    'Activations', 'relu', ...            % ReLU activation
    'IterationLimit', 150, ...            % Number of epochs
    'Verbose', 1, ...                     % Show training progress
    'Standardize', false);                % Already normalized

% Save trained model
save('baseline_nn_model.mat', 'net');

fprintf('\nTraining complete and model saved.\n');

% Evaluate on training data
train_pred = predict(net, X_train_B_norm);
train_accuracy = sum(train_pred == y_train_B) / length(y_train_B);
fprintf('Training accuracy: %.2f%%\n', train_accuracy * 100);