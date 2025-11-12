% test_baseline_model.m
% Evaluate trained model on test data

% Load test data and model
load('normalized_splits.mat', 'X_test_B_norm', 'y_test_B');
load('baseline_nn_model.mat', 'net');

fprintf('Evaluating model on test set...\n');

% Get predictions and probability scores
[predicted_labels, scores] = predict(net, X_test_B_norm);

% Compute accuracy
accuracy = sum(predicted_labels == y_test_B) / length(y_test_B);
fprintf('Test set classification accuracy: %.2f%%\n', accuracy * 100);

% Save predictions
save('test_predictions.mat', 'predicted_labels', 'scores', 'accuracy');

% Display confusion matrix
figure;
confusionchart(y_test_B, predicted_labels);
title('Confusion Matrix - Test Set');

% Display per-class accuracy
for user = 1:10
    user_indices = (y_test_B == user);
    user_accuracy = sum(predicted_labels(user_indices) == user) / sum(user_indices);
    fprintf('User %d accuracy: %.2f%%\n', user, user_accuracy * 100);
end
