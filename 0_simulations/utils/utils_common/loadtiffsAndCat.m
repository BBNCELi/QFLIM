% Load tiffs and concatenate them in given dimension.

%%ELiiiiiii, 20211107
function [out, catLengths] = loadtiffsAndCat(path,namesCell, catDim)
if ~exist('catDim','var') || isempty(catDim)
    catDim = 3;
end
if ~iscell(namesCell)
    error('Error: input a cell struct that contains multiple names');
end

out = [];
catLengths = [];
for count = 1:length(namesCell)
    name = namesCell{count};
    disp(['Loading ', name,' ...']);
    tmp = loadtiff([path,'//',name]);
    out = cat(catDim, out, tmp);
    catLengths = cat(1,catLengths,size(tmp,catDim));
end
