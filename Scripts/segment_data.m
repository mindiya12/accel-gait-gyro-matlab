% ============================================================
% BATCH SEGMENTATION FOR ALL CSV FILES
% This code segments all user data files into 5-second windows
% ============================================================

% --- Settings ---
folderPath = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\Data';          % Path to folder with all CSV files
window_s = 5;                     % Segment size: 5 seconds
sampling_rate = 31;               % Approximate sampling rate (Hz)
window_size = window_s * sampling_rate;  % Number of samples per segment (155)
overlap = 0;                      % No overlap (0 = non-overlapping windows)

% --- Find all CSV files in the folder ---
fileList = dir(fullfile(folderPath, '*.csv'));
numFiles = length(fileList);
fprintf('Found %d CSV files to process.\n', numFiles);

% --- Storage for all segments from all files ---
allSegments = {};     % Will hold all segments from all files
allLabels = [];       % Will hold user labels for each segment

% --- Loop through each file ---
for fileIdx = 1:numFiles
    % Get the current file name
    currentFile = fileList(fileIdx).name;
    filePath = fullfile(folderPath, currentFile);
    
    fprintf('\n--- Processing file %d/%d: %s ---\n', fileIdx, numFiles, currentFile);
    
    try
        % --- Load the data ---
        rawData = readtable(filePath);
        nRows = height(rawData);
        
        % --- Initialize for this file ---
        segments = {};
        segCount = 0;
        step = window_size - overlap;
        
        % --- Segment the data ---
        for startIdx = 1 : step : (nRows - window_size + 1)
            segCount = segCount + 1;
            segments{segCount, 1} = rawData(startIdx : startIdx + window_size - 1, :);
        end
        
        % --- Extract user number from filename ---
        % Example: U1NW_FD.csv -> User 1, U10NW_MD.csv -> User 10
        userNum = str2double(regexp(currentFile, 'U(\d+)', 'tokens', 'once'));
        
        % --- Store segments and labels ---
        allSegments = [allSegments; segments];  % Append to master list
        allLabels = [allLabels; repmat(userNum, segCount, 1)];  % Label each segment
        
        % --- Report progress ---
        fprintf('  -> Created %d segments of %d samples each\n', segCount, window_size);
        fprintf('  -> Assigned to User %d\n', userNum);
        
    catch ME
        % --- Handle errors gracefully ---
        fprintf('  ERROR: Failed to process %s\n', currentFile);
        fprintf('  Reason: %s\n', ME.message);
    end
end

% --- Final Summary ---
fprintf('\n==========================================================\n');
fprintf('SEGMENTATION COMPLETE\n');
fprintf('==========================================================\n');
fprintf('Total files processed: %d\n', numFiles);
fprintf('Total segments created: %d\n', length(allSegments));
fprintf('Unique users: %d\n', length(unique(allLabels)));
fprintf('==========================================================\n');

% --- Save the results ---
save('C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\results/segmented_data.mat', 'allSegments', 'allLabels');
fprintf('Segmented data saved to: ../results/segmented_data.mat\n');
