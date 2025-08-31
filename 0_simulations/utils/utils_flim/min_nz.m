% Calculate the minimum of non-zero elements

%%ELiiiiiii, 20240119
function output = min_nz(input, axis)
%%
if nargin < 2
    axis = 'all';
end

%%
input(input==0) = nan;
output = min(input, [], axis);
output(isnan(output)) = 0;
