% List all folders in "path" that starts with "startString" and ends with "endString".

%%ELiiiiiii, 20250222, modified from function "findeNameList"
function nameList = findNameList_foldersOnly(path,startString,endString)
%% default
if nargin == 1
    startString = [];
    endString = [];
elseif nargin == 2
    endString = [];
end

%%
oldPath = pwd;
cd(path);
result = dir([startString, '*', endString]);
% Filter for directories only
nameList = {result([result.isdir] & ~ismember({result.name}, {'.', '..'})).name}; 
cd(oldPath);


