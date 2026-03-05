function varValue = getVarValue(logsout, varName)
    % 1. Get the timeseries object
    ts = logsout.get(varName).Values;

    % 2. Find the index closest to 0.03 and 0.09
    % This avoids the 'numeric values' error entirely
    [~, idxHealthy] = min(abs(ts.Time - 0.03));
    [~, idxFaulty]  = min(abs(ts.Time - 0.09));

    % 3. Extract the data at those indices
    % squeeze() removes extra dimensions (1x1x1 -> 1)
    hData = squeeze(ts.Data(idxHealthy));
    fData = squeeze(ts.Data(idxFaulty));

    % 4. Return as a 2x1 vector
    varValue = [hData; fData];
end

% function varValue = getVarValue(logsout, varName)
% %GETVARVALUE Summary of this function goes here
% %   Detailed explanation goes here
% healthy = getsampleusingtime(logsout.get(varName).Values, 0.03).Data
% faulty = getsampleusingtime(logsout.get(varName).Values, 0.09).Data
% 
% %varValue = zeros(2, 1)
% 
% varValue(1) = healthy;
% varValue(2) = faulty;
% %varValue = logsout.get(varName).Values.Data(end);
% end

