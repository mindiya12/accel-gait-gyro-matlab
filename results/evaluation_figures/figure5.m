%% FIGURE 5: Grouped Bar Chart - Final Model Comparison
% Purpose: Compare mean EER and std EER for three final models

clear; clc; close all;

% Data from Table 5: Final Model Performance
models = {'Baseline', 'Efficiency', 'Real-World Optimized'};
mean_eer = [2.00, 3.24, 1.14];
std_eer = [3.05, 4.43, 2.12];
features = [133, 20, 80];

% Create figure
figure('Position', [100, 100, 1000, 600], 'Color', 'w');

% Create grouped bar chart
x = 1:3;
bar_width = 0.4;

hold on;

% Mean EER bars
b1 = bar(x - bar_width/2, mean_eer, bar_width, 'FaceColor', [0.2 0.6 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 1.5);

% Std EER bars
b2 = bar(x + bar_width/2, std_eer, bar_width, 'FaceColor', [0.8 0.4 0.2], ...
    'EdgeColor', 'k', 'LineWidth', 1.5);

% Add value labels on bars
for i = 1:length(mean_eer)
    text(x(i) - bar_width/2, mean_eer(i) + 0.2, sprintf('%.2f%%', mean_eer(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(x(i) + bar_width/2, std_eer(i) + 0.2, sprintf('%.2f%%', std_eer(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

% Add feature count labels below bars (WITHOUT FontStyle)
for i = 1:length(features)
    text(x(i), -0.5, sprintf('(%d features)', features(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Highlight recommended model (Real-World Optimized)
rectangle('Position', [2.55, 0, 0.5, 5.2], 'EdgeColor', [0.2 0.8 0.2], ...
    'LineWidth', 4, 'LineStyle', '--');
text(2.8, 4.8, 'SELECTED', 'FontSize', 13, 'FontWeight', 'bold', ...
    'Color', [0.2 0.8 0.2], 'HorizontalAlignment', 'center');

% Formatting
set(gca, 'XTick', x, 'XTickLabel', models, 'FontSize', 13, 'FontWeight', 'bold', 'LineWidth', 1.2);
xlabel('FFMLP Model Configuration', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Error Rate (%)', 'FontSize', 15, 'FontWeight', 'bold');
title('Figure 5. Comparison of Final FFMLP Models (Cross-Day Evaluation)', ...
    'FontSize', 16, 'FontWeight', 'bold');
legend({'Mean EER', 'Std EER'}, 'Location', 'northwest', 'FontSize', 12);
grid on;
grid minor;
ylim([-0.8, max(std_eer) * 1.25]);

hold off;

% Save figure
saveas(gcf, 'Figure5_Model_Comparison.png');
saveas(gcf, 'Figure5_Model_Comparison.fig');
print(gcf, 'Figure5_Model_Comparison.png', '-dpng', '-r300');

fprintf('✓ Figure 5 generated and saved successfully!\n');
fprintf('  - Figure5_Model_Comparison.png (300 DPI)\n');
fprintf('  - Figure5_Model_Comparison.fig (editable)\n\n');
