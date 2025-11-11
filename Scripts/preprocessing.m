% Folder where your CSVs live
folderPath = 'C:\Users\E-TECH\OneDrive - NSBM\Desktop\ai ml\accel-gait-gyro-matlab\Data';
% Find all CSV files in that folder
fileList = dir(fullfile(folderPath, '*.csv'));

% Counter to track success
successCount = 0;
totalFiles = length(fileList);

fprintf('Starting to process %d files...\n\n', totalFiles);

for k = 1:totalFiles
    % Build full filename
    filename = fullfile(folderPath, fileList(k).name);
    
    try
        % --- Your data loading and segmenting code goes here ---
        data = readtable(filename);
        % ...segment, extract features, etc...
        
        % If processing completes without error, count it as success
        successCount = successCount + 1;
        fprintf('✓ File %d/%d processed: %s\n', k, totalFiles, fileList(k).name);
        
    catch ME
        % If error occurs, show which file failed
        fprintf('✗ File %d/%d FAILED: %s - Error: %s\n', k, totalFiles, fileList(k).name, ME.message);
    end
end

% Final summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Total files: %d\n', totalFiles);
fprintf('Successfully processed: %d\n', successCount);
fprintf('Failed: %d\n', totalFiles - successCount);

if successCount == totalFiles
    fprintf('✓ ALL FILES PROCESSED SUCCESSFULLY!\n');
else
    fprintf('✗ Some files failed. Check errors above.\n');
end
fprintf('=============================\n');
