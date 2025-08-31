%% save input data as .tiff or .tif file %%
%% ELiiiiiii, 20210426
function res = saveastiff_colorDepth(data, path, options)
% inputs:
%     data: 3D stack or 4D stack frames
%         if data is a 3D stack, i.e. xdim * ydim * zdim
%             then encode zdim into color and save a 2D image, i.e. xdim * ydim (* color)
%         if data is a 4D stack frames, i.e. xdim * ydim * zdim * frames
%             then encode zdim into color and save a video, i.e. xdim * ydim (* color) * frames
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
end

%% self-defined time display _______ start
if ~isfield(options, 'message'),   options.message   = true; end
optionsInput.message = options.message;
options.message = false;
tStart = tic;


%% color-depth coding
zdim=size(data,3);
colorMap = jet(zdim);%%!!!

%%version1: slow
% data_colorDepth=zeros(size(data,1),size(data,2),3,size(data,4),'like',data);
% for z=1:zdim
%     for color=1:3
%         data_colorDepth(:,:,color,:)=data_colorDepth(:,:,color,:)+data(:,:,z,:).*colorMap(z,color);
%     end
% end
%%version2: about 5 times faster
data_colorDepth=zeros(size(data,1),size(data,2),3,size(data,4),'like',data);
for color=1:3
    data_colorDepth(:,:,color,:)=sum(data.*permute(colorMap(:,color),[3,4,1,2]),3);
end


options.color = true;
res = saveastiff(data_colorDepth, path, options);


%% self-defined time display _______ finish
tElapsed = toc(tStart);
if optionsInput.message
    fprintf('The color-depth coded file was saved successfully. Elapsed time : %.3f s.\n', tElapsed);
end