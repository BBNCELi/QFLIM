% save input hyperstack with RGB the last dimension
% as .tiff or .tif

%%% ELiiiiiii, 20250315
function res = saveastiff_RGBAsLastDim(data, path, options)
% inputs:
%     data: 3D or 4D images with RGB as the last dimension
%     path: file name
%     options: saving settings
%         default:
%         options.compress  = 'no';
%         options.message   = true;
%         options.append    = false;
%         options.overwrite = true;
%         options.big       = false;
ndim = ndims(data);
size1 = size(data, 1);
size2 = size(data, 2);
sizec = size(data, ndim);
if (sizec ~= 3) && (sizec ~= 4)
    error('The last dimension should be RGB or CMYK. Use original saveastiff otherwise');
end
if nargin < 3 % Use default options
    options.compress = 'no';
    options.message = true;
    options.append = false;
    options.overwrite = true;
end
options.color = true;

%% self-defined time display _______ start
if ~isfield(options, 'message'),   options.message   = true; end
optionsInput.message = options.message;
options.message = false;
tStart = tic;

%% permute
if ndim == 2
    data = permute(data, [1,3,2]);
elseif ndim == 3
%     data = permute(data, [1,2,4,3]);
%     data = permute(data, [1,2,4,3]);
    data = reshape(data, [size(data), 1]);
elseif ndim == 4
    data = permute(data, [1,2,4,3]);
elseif ndim > 4
    data = reshape(data, size1, size2, [], sizec);
    data = permute(data, [1, 2, 4, 3]);
end
res = saveastiff(data, path, options);

%% self-defined time display _______ finish
tElapsed = toc(tStart);
if optionsInput.message
    fprintf('Color-coded tiff saved successfully. Elapsed time : %.3f s.\n', tElapsed);
end