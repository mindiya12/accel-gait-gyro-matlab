% Folder paths
inputFolder = 'D:\src\development\accel-gait-gyro-matlab\Data';       % Input folder with your raw CSVs
outputFolder = 'D:\src\development\accel-gait-gyro-matlab\Data\cleaned'; % Folder to save cleaned CSVs

% Create output folder if not exists
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of CSV files
files = dir(fullfile(inputFolder, '*.csv'));

for k = 1:length(files)
    % Read one file
    filename = fullfile(inputFolder, files(k).name);
    rawData = readtable(filename);

    % --- Remove or impute missing data ---
    missing_any = any(ismissing(rawData), 'all');
    if missing_any
        rawData = fillmissing(rawData, 'linear');
    end

    % --- Remove obvious outliers column-wise ---
    for c = 1:width(rawData)
        col = rawData{:, c};
        % Identify outliers based on mean and std deviation
        mu = mean(col, 'omitnan');
        sigma = std(col, 'omitnan');
        lowerBound = mu - 3*sigma;
        upperBound = mu + 3*sigma;
        outlierIdx = (col < lowerBound) | (col > upperBound);
        % Interpolate over outliers
        validIdx = find(~outlierIdx);
        outlierIdxs = find(outlierIdx);
        if length(validIdx) > 1
            col(outlierIdxs) = interp1(validIdx, col(validIdx), outlierIdxs, 'linear', 'extrap');
        else
            % If too few valid points, replace with mean
            col(outlierIdxs) = mu;
        end
        rawData{:, c} = col;
    end

    % Save the cleaned table
    outFilename = fullfile(outputFolder, files(k).name);
    writetable(rawData, outFilename);

    fprintf('Cleaned and saved file %d/%d: %s\n', k, length(files), files(k).name);
end

fprintf('All files cleaned successfully and saved to %s\n', outputFolder);
