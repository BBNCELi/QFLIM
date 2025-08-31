%%Simple function that judge if a input number is in a certain (closed) interval or not.
%%If yes, return the interval. If no, return the bound it exceeds.

%% ELiiiiiii, 20210818
function [output, isIn] = cropNum(input, lowerBound, upperBound)
%%
    if nargin < 3
        upperBound = inf;
    end
    if nargin <2
        lowerBound = -inf;
    end
    if upperBound < lowerBound
        error('Incorrect interval!');
    end
%%
    if input < lowerBound
        output = lowerBound;
        isIn = false;
        return;
    elseif input > upperBound
        output = upperBound;
        isIn = false;
        return;
    else
        output = input;
        isIn = true;
        return;
    end
end

