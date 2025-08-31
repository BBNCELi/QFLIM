%% save input hyperstack as .tiff or .tif stack %%
%% ELiiiiiii, 20240620
function res = saveastiff_hyperstack(data, path, options)
% inputs:
%     data: 4D or higher dimensional hyperstack
%         This function flattens dimensions >= 3 and save it as a stack
%     path: file name
%     options: saving settings
%         default:
%         options.compress  = 'no';
%         options.message   = true;
%         options.append    = false;
%         options.overwrite = false;
%         options.big       = false;
if nargin < 3 % Use default options
    options.compress = 'no';
    options.message = true;
    options.append = false;
    options.overwrite = false;
    options.color = false;
end

%% self-defined time display _______ start
if ~isfield(options, 'message'),   options.message   = true; end
optionsInput.message = options.message;
options.message = false;
tStart = tic;

%% high dimensions flattening
sizex = size(data, 1);
sizey = size(data, 2);

data_flat = reshape(data, sizex, sizey, []);

res = saveastiff(data_flat, path, options);


%% self-defined time display _______ finish
tElapsed = toc(tStart);
if optionsInput.message
    fprintf('Hyperstack saved successfully. Elapsed time : %.3f s.\n', tElapsed);
end