%% FIGURE 1: Bar Chart of EER (%) per Scenario with Error Bars
% Purpose: Compare baseline performance across same-day, cross-day, and combined scenarios

clear; clc; close all;

% Data from Table 1: Baseline Results
scenarios = {'Same-Day', 'Cross-Day', 'Combined-Day'};
mean_eer = [0.00, 1.25, 0.00];  % Mean EER (%)
std_eer = [0.00, 1.75, 0.00];   % Standard deviation (%)

% Create figure
figure('Position', [100, 100, 900, 600], 'Color', 'w');
hold on;

% Create bar chart
b = bar(mean_eer, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', [0 0 0], 'LineWidth', 1.5);

% Add error bars
errorbar(1:length(mean_eer), mean_eer, std_eer, 'LineStyle', 'none', ...
    'Color', 'k', 'LineWidth', 2, 'CapSize', 15);

% Formatting
set(gca, 'XTickLabel', scenarios, 'FontSize', 13, 'FontWeight', 'bold', 'LineWidth', 1.2);
xlabel('Partitioning Scenario', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Equal Error Rate (%)', 'FontSize', 15, 'FontWeight', 'bold');
title('Figure 1. Baseline EER Across Different Partitioning Scenarios', ...
    'FontSize', 16, 'FontWeight', 'bold');
grid on;
grid minor;
ylim([0, max(mean_eer + std_eer) * 1.4]);

% Add value labels on bars
for i = 1:length(mean_eer)
    if mean_eer(i) > 0
        text(i, mean_eer(i) + std_eer(i) + 0.2, ...
            sprintf('%.2f%% ± %.2f%%', mean_eer(i), std_eer(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    else
        text(i, 0.2, '0.00%', 'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
end

hold off;

% Save figure
saveas(gcf, 'Figure1_EER_Scenarios.png');
saveas(gcf, 'Figure1_EER_Scenarios.fig');
print(gcf, 'Figure1_EER_Scenarios.png', '-dpng', '-r300');

fprintf('✓ Figure 1 generated and saved successfully!\n');
fprintf('  - Figure1_EER_Scenarios.png (300 DPI)\n');
fprintf('  - Figure1_EER_Scenarios.fig (editable)\n\n');
