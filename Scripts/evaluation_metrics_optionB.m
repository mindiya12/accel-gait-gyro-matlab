% --- Variables you must already have ---
% y_test_B: Nx1, true labels
% y_pred_B: 10xN, NN softmax outputs for each sample
% y_pred_B_class: Nx1, predicted class for each sample

thresholds = 0:0.01:1;
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));
N = numel(y_test_B);
scores = max(y_pred_B, [], 1)'; % Highest confidence per sample (1xN -> N x1)

for ti = 1:length(thresholds)
    th = thresholds(ti);
    false_accepts = 0;
    false_rejects = 0;
    total_impostor = 0;
    total_genuine = 0;
    for i = 1:N
        pred_label = y_pred_B_class(i);
        true_label = y_test_B(i);
        score_i = scores(i);
        % If network predicts correctly (genuine attempt)
        if pred_label == true_label
            total_genuine = total_genuine + 1;
            if score_i < th
                false_rejects = false_rejects + 1;
            end
        else % Impostor trial (wrong user claimed)
            total_impostor = total_impostor + 1;
            if score_i >= th
                false_accepts = false_accepts + 1;
            end
        end
    end
    FRR(ti) = false_rejects / max(1, total_genuine);
    FAR(ti) = false_accepts / max(1, total_impostor);
end

% --- Print results at key thresholds, especially EER ---
diff = abs(FAR - FRR);
[~, min_idx] = min(diff);
EER = mean([FAR(min_idx), FRR(min_idx)]);
threshold_EER = thresholds(min_idx);
fprintf('Equal Error Rate (EER): %.2f%% at threshold=%.2f\n', EER*100, threshold_EER);

fprintf('At threshold %.2f: FAR = %.2f%%, FRR = %.2f%%\n', threshold_EER, FAR(min_idx)*100, FRR(min_idx)*100);

% Print FAR/FRR at low, medium, high thresholds for context
idx_low = 1; idx_mid = round(length(thresholds)/2); idx_high = length(thresholds);
fprintf('At low threshold %.2f: FAR = %.2f%%, FRR = %.2f%%\n', thresholds(idx_low), FAR(idx_low)*100, FRR(idx_low)*100);
fprintf('At medium threshold %.2f: FAR = %.2f%%, FRR = %.2f%%\n', thresholds(idx_mid), FAR(idx_mid)*100, FRR(idx_mid)*100);
fprintf('At high threshold %.2f: FAR = %.2f%%, FRR = %.2f%%\n', thresholds(idx_high), FAR(idx_high)*100, FRR(idx_high)*100);

% --- Plot FAR / FRR / EER ---
figure;
plot(thresholds, FAR*100, 'r', 'LineWidth', 2); hold on;
plot(thresholds, FRR*100, 'b', 'LineWidth', 2);
plot(threshold_EER, EER*100, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
legend('FAR (%)','FRR (%)','EER threshold');
xlabel('Decision Threshold'); ylabel('Error Rate (%)'); title('FAR/FRR/EER vs Threshold'); grid on;
