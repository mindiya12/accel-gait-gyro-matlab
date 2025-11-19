%% FIGURE 2: Per-User EER Distribution (Without Statistics Toolbox)
% Purpose: Show fairness and per-user performance variation

clear; clc; close all;

% Data from Table 2: Per-User Metrics (Cross-Day)
user_ids = 1:10;
per_user_eer = [0.0, 0.5, 0.9, 2.4, 0.7, 2.6, 1.9, 3.1, 0.5, 1.5];

% Statistics
mean_eer = mean(per_user_eer);
median_eer = median(per_user_eer);
std_eer = std(per_user_eer);
q1 = prctile(per_user_eer, 25);  % First quartile
q3 = prctile(per_user_eer, 75);  % Third quartile
min_eer = min(per_user_eer);
max_eer = max(per_user_eer);

% Create figure
figure('Position', [100, 100, 900, 600], 'Color', 'w');
hold on;

% Manual boxplot components
box_width = 0.4;
x_center = 1;

% Draw box (Q1 to Q3)
rectangle('Position', [x_center - box_width/2, q1, box_width, q3-q1], ...
    'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 2);

% Draw median line
plot([x_center - box_width/2, x_center + box_width/2], [median_eer, median_eer], ...
    '-', 'Color', 'r', 'LineWidth', 3);

% Draw whiskers
plot([x_center, x_center], [q3, max_eer], '-k', 'LineWidth', 2);
plot([x_center, x_center], [q1, min_eer], '-k', 'LineWidth', 2);
plot([x_center - box_width/4, x_center + box_width/4], [max_eer, max_eer], '-k', 'LineWidth', 2);
plot([x_center - box_width/4, x_center + box_width/4], [min_eer, min_eer], '-k', 'LineWidth', 2);

% Scatter individual points with slight jitter for visibility
jitter = 0.08 * randn(size(per_user_eer));
scatter(ones(size(per_user_eer)) + jitter, per_user_eer, 120, ...
    'filled', 'MarkerFaceColor', [0.8 0.3 0.3], ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.3, 'MarkerFaceAlpha', 0.7);

% Add mean line (dashed green)
plot([x_center - box_width/2 - 0.15, x_center + box_width/2 + 0.15], ...
    [mean_eer, mean_eer], '--', 'Color', [0.2 0.8 0.2], 'LineWidth', 3);

% Formatting
xlim([0.4, 1.6]);
ylim([-0.3, max_eer * 1.2]);
ylabel('Equal Error Rate (%)', 'FontSize', 15, 'FontWeight', 'bold');
title('Figure 2. Distribution of Per-User EER (Cross-Day Scenario)', ...
    'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'XTick', x_center, 'XTickLabel', {'All 10 Users'}, ...
    'FontSize', 13, 'FontWeight', 'bold', 'LineWidth', 1.2);
grid on;

% Add statistics annotation box
dim = [0.15 0.65 0.28 0.25];
str = {sprintf('\\bfStatistics:'), ...
       sprintf('Mean: %.2f%%', mean_eer), ...
       sprintf('Median: %.2f%%', median_eer), ...
       sprintf('Std Dev: %.2f%%', std_eer), ...
       sprintf('Q1: %.2f%%, Q3: %.2f%%', q1, q3), ...
       sprintf('Min: %.2f%%, Max: %.2f%%', min_eer, max_eer)};
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', 'LineWidth', 1.5, ...
    'FontSize', 10);

% Legend
legend({'Box (Q1-Q3)', 'Median', '', '', '', '', '', '', '', '', ...
    '', '', 'Individual Users', 'Mean EER'}, ...
    'Location', 'northeast', 'FontSize', 11);

hold off;

% Save figure
saveas(gcf, 'Figure2_PerUser_Distribution.png');
saveas(gcf, 'Figure2_PerUser_Distribution.fig');
print(gcf, 'Figure2_PerUser_Distribution.png', '-dpng', '-r300');

fprintf('✓ Figure 2 generated and saved successfully!\n');
fprintf('  - Figure2_PerUser_Distribution.png (300 DPI)\n');
fprintf('  - Figure2_PerUser_Distribution.fig (editable)\n\n');
