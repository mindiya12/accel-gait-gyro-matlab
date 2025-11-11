% ============================================================
% BATCH SEGMENTATION FOR ALL CSV FILES - COMPATIBLE VERSION
% Works with all MATLAB versions
% ============================================================

% --- Settings ---
folderPath = 'C:\Users\Welcome\OneDrive - NSBM\Desktop\3rd_year\ai_ml\Corsework\accel-gait-gyro-matlab\Data';
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
        % --- Load the data as matrix first ---
        rawMatrix = readmatrix(filePath);
        
        % Check dimensions
        [nRows, nCols] = size(rawMatrix);
        fprintf('  -> File has %d rows and %d columns\n', nRows, nCols);
        
        % --- Define column names based on number of columns ---
        if nCols == 7
            % First column is index/timestamp, columns 2-7 are sensors
            colNames = {'Time', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'};
            fprintf('  -> Detected 7 columns (Time + 6 sensors)\n');
        elseif nCols == 6
            % All columns are sensors
            colNames = {'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'};
            fprintf('  -> Detected 6 columns (sensors only)\n');
        else
            error('Unexpected number of columns: %d. Expected 6 or 7.', nCols);
        end
        
        % Convert matrix to table with proper column names
        rawData = array2table(rawMatrix, 'VariableNames', colNames);
        
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
        fprintf('  -> SUCCESS!\n');
        
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

% Check if we got any segments
if isempty(allSegments)
    error('No segments were created! Check your data files.');
end

% --- Verify first segment structure ---
fprintf('\nFirst segment verification:\n');
fprintf('  Size: %d rows × %d columns\n', height(allSegments{1}), width(allSegments{1}));
fprintf('  Columns: %s\n', strjoin(allSegments{1}.Properties.VariableNames, ', '));

% --- Save the results ---
savePath = fullfile(folderPath, '..', 'results', 'segmented_data.mat');
save(savePath, 'allSegments', 'allLabels');
fprintf('\nSegmented data saved to: %s\n', savePath);