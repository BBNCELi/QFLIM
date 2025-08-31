% Find min value in a cell. Ignoring no-number elements.

%%ELiiiiiii, 20211110
function maxValue = minInCell(inputCell)
inputCell = inputCell(:);
allNums = [];
for i = 1:length(inputCell)
    cellHere = inputCell{i};
    if isnumeric(cellHere)
        allNums = [allNums; cellHere(:)];
    end
end
maxValue = min(allNums);
