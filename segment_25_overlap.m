% ============================================================
% CREATE 25% OVERLAPPING SEGMENTS
% ============================================================

% --- Settings ---
folderPath = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\Data';
window_s = 5;
sampling_rate = 31;
window_size = window_s * sampling_rate;  % 155 samples

% --- 25% OVERLAP CONFIGURATION ---
overlap_percent = 25;
overlap_samples = round(window_size * overlap_percent / 100);  % ~39 samples
step_size = window_size - overlap_samples;  % ~116 samples

fprintf('=== CREATING 25%% OVERLAPPING SEGMENTS ===\n');
fprintf('Window: %d samples (%.1f sec)\n', window_size, window_s);
fprintf('Overlap: %d%% (%d samples)\n', overlap_percent, overlap_samples);
fprintf('Step size: %d samples\n\n', step_size);

% --- Find CSV files ---
fileList = dir(fullfile(folderPath, '*.csv'));
numFiles = length(fileList);
fprintf('Found %d CSV files.\n\n', numFiles);

% --- Storage ---
allSegments = {};
allLabels = [];

% --- Process each file ---
for fileIdx = 1:numFiles
    currentFile = fileList(fileIdx).name;
    filePath = fullfile(folderPath, currentFile);
    
    fprintf('Processing %d/%d: %s\n', fileIdx, numFiles, currentFile);
    
    try
        rawMatrix = readmatrix(filePath);
        [nRows, nCols] = size(rawMatrix);
        
        % Define column names
        if nCols == 7
            colNames = {'Time', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'};
        elseif nCols == 6
            colNames = {'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'};
        else
            error('Unexpected columns: %d', nCols);
        end
        
        rawData = array2table(rawMatrix, 'VariableNames', colNames);
        
        % Create overlapping segments
        segments = {};
        segCount = 0;
        
        for startIdx = 1 : step_size : (nRows - window_size + 1)
            segCount = segCount + 1;
            endIdx = startIdx + window_size - 1;
            segments{segCount, 1} = rawData(startIdx:endIdx, :);
        end
        
        % Extract user ID
        userNum = str2double(regexp(currentFile, 'U(\d+)', 'tokens', 'once'));
        
        if isnan(userNum)
            warning('Cannot extract user ID from %s', currentFile);
            continue;
        end
        
        % Store
        allSegments = [allSegments; segments];
        allLabels = [allLabels; repmat(userNum, segCount, 1)];
        
        fprintf('  Created %d segments for User %d\n', segCount, userNum);
        
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end

% --- Summary ---
fprintf('\n=== SUMMARY ===\n');
fprintf('Total segments: %d\n', length(allSegments));
fprintf('Users: %d\n', length(unique(allLabels)));

% --- Save ---
resultsFolder = fullfile(folderPath, '..', 'results');
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

saveFileName = 'segmented_data_overlap25.mat';
savePath = fullfile(resultsFolder, saveFileName);

segmentation_config = struct(...
    'window_size', window_size, ...
    'overlap_percent', overlap_percent, ...
    'overlap_samples', overlap_samples, ...
    'step_size', step_size);

save(savePath, 'allSegments', 'allLabels', 'segmentation_config');
fprintf('\nSaved to: %s\n', savePath);
