% Choose a non-zero component randomly along given axis

%%ELiiiiiii, 20240226: This function was written by chatGPT. REALLY slow
function output = chooseOne_nz(input, axis)
    % Choose a non-zero component randomly along the specified axis of an n-dimensional array
    
    % Validate input
    if axis < 1 || axis > ndims(input)
        error('Invalid axis value.');
    end

    % Initialize the size for the output
    sz = size(input);
    outputSize = sz;
    outputSize(axis) = 1; % Set the size along the specified axis to 1
    output = NaN(outputSize); % Initialize output with NaNs
    
    % Generate a grid of indices for all axes except the specified one
    ind = repmat({':'}, 1, ndims(input));
    
    % Iterate over every position in the output array
    for i = 1:prod(outputSize)
        [ind{:}] = ind2sub(outputSize, i); % Convert linear index to subscript indices
        slice = ind;
        slice{axis} = ':'; % Select the entire slice along the specified axis
        
        % Extract the non-zero elements from the current slice of the input
        currentSlice = input(slice{:});
        nonZeroElements = currentSlice(currentSlice ~= 0);
        
        if ~isempty(nonZeroElements)
            % Randomly choose one of the non-zero elements
            randIndex = randi(length(nonZeroElements), 1);
            selectedValue = nonZeroElements(randIndex);
            
            % Assign the selected value to the output
            output(ind{:}) = selectedValue;
        else
            % If there are no non-zero elements, leave the output as NaN
            output(ind{:}) = NaN;
        end
    end
end