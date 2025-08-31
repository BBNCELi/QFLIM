% save input hyperstack with RGB the last dimension
% as .tiff or .tif

%% ELiiiiiii, 20250707, use imwrite instead of saveastiff
function saveastiff_RGBAsLastDim_imwrite(data, path)
% inputs:
%     data: 3D or 4D images with RGB as the last dimension
%     path: file name

    if ndims(data) == 3; data = permute(data, [1,2,4,3]); end
    [H, W, N, C] = size(data);
    if C ~= 3
        error('Last dimension must be 3 (RGB)');
    end
    if exist(path, 'file'); delete(path); end

    t = Tiff(path, 'w');

    tagstruct.ImageLength = H;
    tagstruct.ImageWidth = W;
    tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
    tagstruct.Photometric = Tiff.Photometric.RGB;
    tagstruct.BitsPerSample = 8;  % or 16, if uint16
    tagstruct.SamplesPerPixel = 3;
    tagstruct.RowsPerStrip = H;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;

    for i = 1:N
        t.setTag(tagstruct);
        frame = squeeze(data(:,:,i,:));
        t.write(frame);
        if i < N
            t.writeDirectory();
        end
    end
    t.close();
end