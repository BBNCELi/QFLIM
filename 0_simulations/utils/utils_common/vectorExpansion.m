% Expand vector to get smoother value change

%%ELi, 20211110
function output = vectorExpansion(input, ratio)
% input:
%    input: input vector, length N
%    ratio: expanding ratio (default 100)
% output:
%     output: output vector, length N*ratio+1

% e.g. input = [1,2,3], ratio = 2, then output = [1, 1.5, 2, 2.5, 3]
% e.g. input = [1,2,3], ratio = 4, then output = [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]

if ~isvector(input)
    error('vectorExpansion ERROR: input must be a vector');
end
if nargin < 2
    ratio = 100;
end

output = [];
for inputCount = 1:length(input)-1
    subVec = linspace(input(inputCount),input(inputCount+1),ratio+1);
    if inputCount == 1
        output = [output,subVec(1)];
    end
    output = [output, subVec(2:end)];
end