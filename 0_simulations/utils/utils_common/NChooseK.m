%Select K members from N members, returning min(all, maxOptionNum) possibilities
%%ELiiiiiii, 20211207
%%ELiiiiiii, 20211208, add maxOptionNum
function [index_chosen, index_notChosen, optionNum] = NChooseK(N,K,maxOptionNum)
% inputs:
%     N: scalar
%     K: scalar
%     maxOptionNum: in case when there are too many posibilities
% outputs:
%     index_chosen: matrix, optionNum * K, where optionNum = nchoosek(N,K)
%     index_notChosen: matrix, optionNum * (N - K), where optionNum = nchoosek(N,K)
if ~isscalar(N)
    error('Input error: input a scalar for N');
end
if nargin < 3
    maxOptionNum = inf;
end
%%
maybeOptionNum = nchoosek(N,K);
if maybeOptionNum > maxOptionNum
    optionNum = maxOptionNum;
    index_chosen = [];
    while size(index_chosen,1) < maxOptionNum
        index_chosen = [index_chosen; randperm(N,K)];
        index_chosen = unique(index_chosen,'rows');
    end
else
    optionNum = maybeOptionNum;
    index_chosen =  nchoosek(1:N,K);
end

index_notChosen = zeros(optionNum, N - K);
for i = 1:optionNum
    index_notChosen(i,:) = setdiff(1:N,index_chosen(i,:));
end





