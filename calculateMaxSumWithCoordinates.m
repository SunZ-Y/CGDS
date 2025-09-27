function [maxSum, rowIndices] = calculateMaxSumWithCoordinates(data)
    [rows, cols] = size(data);
    
    % 找到整个矩阵的最大值的索引
    [maxRow, ~] = find(data == max(data(:)), 1, 'first');
    
    dp = zeros(rows, cols); % 存储从起始位置到第 i 列，选择第 i 列的最大和
    chosenIndices = zeros(rows, cols); % 记录选择的列的索引

    % 初始条件
    dp(maxRow, 1) = data(maxRow, 1);

    % 动态规划计算
    for j = 2:cols
        for i = 1:rows
            start_row = max(1, i-1);
            end_row = min(rows, i+1);
            [maxVal, maxIdx] = max(dp(start_row:end_row, j-1) - 3e7 * abs((start_row:end_row)' - i)); % 减去相邻数据纵坐标的差
            dp(i, j) = maxVal + data(i, j);
            chosenIndices(i, j) = start_row - 1 + maxIdx; % 记录选择的列的索引
        end
    end

    % 找到最大和
    maxSum = max(dp(:));

    % 回溯找到每一列的索引
    rowIndices = zeros(1, cols);
    [~, ~] = find(dp == maxSum, 1, 'first');
    rowIndices(end) = maxRow;
    for j = cols-1:-1:1
        rowIndices(j) = chosenIndices(rowIndices(j+1), j+1);
    end
end