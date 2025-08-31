% Find max value in a cell. Ignoring no-number elements.

%%ELiiiiiii, 20211110
function maxValue = maxInCell(inputCell)
inputCell = inputCell(:);
allNums = [];
for i = 1:length(inputCell)
    cellHere = inputCell{i};
    if isnumeric(cellHere)
        allNums = [allNums; cellHere(:)];
    end
end
maxValue = max(allNums);
