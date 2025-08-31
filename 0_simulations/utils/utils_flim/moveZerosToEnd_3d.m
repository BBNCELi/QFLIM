% This function move the 0 elements to the back of the 3d matrix
%%ELiiiiiii, 20241112

function [output, output_truncate] = moveZerosToEnd_3d(input)
%% sizes and initializations
[M, N, P] = size(input);
output = zeros(M, N, P, 'like', input);
maxNonZeroIndex = zeros(M, N);

%% iterate through each pixel
for i = 1:M
    for j = 1:N
        % get the current pixel's values along the third dimension
        pixelValues = squeeze(input(i, j, :));
        
        % Separate non-zero and zero values
        nonZeroValues = pixelValues(pixelValues ~= 0);
        zeroValues = pixelValues(pixelValues == 0);
        maxNonZeroIndex(i,j) = length(nonZeroValues(:));

        % Concatenate non-zero values and zero values
        sortedValues = [nonZeroValues; zeroValues];

        % Assign back
        output(i, j, :) = sortedValues;
    end
end

maxLength = max(maxNonZeroIndex(:));
output_truncate = output(:, :, 1:maxLength);


