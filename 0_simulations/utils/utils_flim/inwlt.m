% Return the intensity-weighted-lifetime version of the input

%%ELiiiiiii, 20250222
%%ELiiiiiii, 20250630, add flag for inwlt_white
function inwlt_out = inwlt(lt_in, in_in, lt_contrast, in_contrast, colormap, flag_white)
%% input
%     lt_in: input lifetime values
%     in_in: input intensity values. size(in_in) == size(lt_in)
%     colormap: 256x3, 0-1 double
%     flag_white: linear interp to black([0,0,0]) or white([255,255,255])

%% check
if nargin < 2; in_in = lt_in; end
if nargin < 3; lt_contrast = [0, max(lt_in(:))]; end
if nargin < 4; in_contrast = [0, max(in_in(:))]; end
if nargin < 5; colormap = jet; end
if nargin < 6; flag_white = false; end
if size(lt_in) ~= size(in_in)
    error('Size mismatch...');
end
size_in = size(lt_in);
ndims_in = ndims(lt_in);
lt_in = lt_in(:);
in_in = in_in(:);
if ~isequal(size(colormap), [256, 3]) || max(colormap(:)) > 1
    error('The input color map should be a 256x3 double matrix');
end

%% rescale
lt_rescale = rescale(lt_in, 0, 1,  'InputMin', lt_contrast(1), 'InputMax', lt_contrast(2));
lt_rescale = uint8(255 * lt_rescale);
lt_rescale_indices = 1 + lt_rescale;
in_rescale = rescale(in_in, 0, 1,  'InputMin', in_contrast(1), 'InputMax', in_contrast(2));

colorMapped = colormap(lt_rescale_indices, :); % Apply colormap
if ~flag_white
    inwlt_r = uint8(255 .* colorMapped(:, 1) .* in_rescale);
    inwlt_g = uint8(255 .* colorMapped(:, 2) .* in_rescale);
    inwlt_b = uint8(255 .* colorMapped(:, 3) .* in_rescale);
else
    inwlt_r = uint8(255 .* (in_rescale .* colorMapped(:, 1) + (1 - in_rescale) .* 1));
    inwlt_g = uint8(255 .* (in_rescale .* colorMapped(:, 2) + (1 - in_rescale) .* 1));
    inwlt_b = uint8(255 .* (in_rescale .* colorMapped(:, 3) + (1 - in_rescale) .* 1));
end

inwlt_out = cat(2, inwlt_r, inwlt_g, inwlt_b);

inwlt_out = reshape(inwlt_out, [size_in, 3]);
