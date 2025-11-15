% ============================================================
% REMOVE DUPLICATE SEGMENTS FROM ALL OVERLAPPING CONFIGURATIONS
% Maintains consistency with baseline model preprocessing
% ============================================================

clear; clc;

fprintf('=== REMOVING DUPLICATE SEGMENTS FROM ALL DATASETS ===\n\n');

resultsFolder = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results';

% --- Define configurations to process ---
configs = {
    'FeatureMatrix_NonOverlapping.mat', 'FeatureMatrix_NonOverlapping_NoDuplicates.mat', 'Non-Overlapping (0%)';
    'FeatureMatrix_Overlap25.mat', 'FeatureMatrix_Overlap25_NoDuplicates.mat', '25% Overlap';
    'FeatureMatrix_Overlap50.mat', 'FeatureMatrix_Overlap50_NoDuplicates.mat', '50% Overlap'
};

% --- Process each configuration ---
for configIdx = 1:size(configs, 1)
    inputFile = configs{configIdx, 1};
    outputFile = configs{configIdx, 2};
    configName = configs{configIdx, 3};
    
    fprintf('==========================================================\n');
    fprintf('Processing: %s\n', configName);
    fprintf('==========================================================\n');
    
    % Load feature matrix
    inputPath = fullfile(resultsFolder, inputFile);
    
    if ~exist(inputPath, 'file')
        warning('File not found: %s. Skipping.', inputPath);
        continue;
    end
    
    load(inputPath); % Loads: featureMatrix, allLabels
    
    originalCount = size(featureMatrix, 1);
    fprintf('Original segments: %d\n', originalCount);
    
    % --- Find unique rows (remove exact duplicates) ---
    [uniqueFeatures, uniqueIdx, ~] = unique(featureMatrix, 'rows', 'stable');
    uniqueLabels = allLabels(uniqueIdx);
    
    duplicatesRemoved = originalCount - size(uniqueFeatures, 1);
    
    fprintf('Duplicates removed: %d\n', duplicatesRemoved);
    fprintf('Remaining segments: %d\n', size(uniqueFeatures, 1));
    fprintf('Reduction: %.2f%%\n', (duplicatesRemoved/originalCount)*100);
    
    % Display distribution per user after duplicate removal
    fprintf('\nSegments per user after duplicate removal:\n');
    uniqueUsers = unique(uniqueLabels);
    for u = 1:length(uniqueUsers)
        userId = uniqueUsers(u);
        userCount = sum(uniqueLabels == userId);
        fprintf('  User %2d: %d segments\n', userId, userCount);
    end
    
    % --- Save cleaned dataset ---
    featureMatrix = uniqueFeatures;
    allLabels = uniqueLabels;
    
    outputPath = fullfile(resultsFolder, outputFile);
    save(outputPath, 'featureMatrix', 'allLabels');
    
    fprintf('\nCleaned dataset saved to: %s\n\n', outputFile);
end

fprintf('==========================================================\n');
fprintf('ALL DATASETS CLEANED SUCCESSFULLY\n');
fprintf('==========================================================\n');
