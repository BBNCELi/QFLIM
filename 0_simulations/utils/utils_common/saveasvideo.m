% A generalized function for saving videos in different formats

%%ELiiiiiii, 20240913
function saveasvideo(inputvideo, savename, format, frameRate)
% Inputs
% inputvideo: uint8
%   size1 x size2 x 3 x frames or size1 x size2 x frames

%% defaults
if ~isa(inputvideo, 'uint8')
    error('Input data must be of type uint8.');
end
if ndims(inputvideo) < 4 %% 2d or 3d input
    inputvideo = repmat(permute(inputvideo, [1, 2, 4, 3]), [1, 1, 3, 1]);
elseif size(inputvideo, 3) == 1
    inputvideo = repmat(inputvideo, [1, 1, 3, 1]);
end

if ~exist('savename', 'var'); savename = ['video_', datestr(now, 'YYYYmmDD_HHMMSS'), '.mp4']; end
if ~exist('format', 'var'); format = savename(end-3:end); end
if ~exist('frameRate', 'var'); frameRate = 30; end

%% create videowrite and settings
switch format
    case '.avi'
%         outputVideo = VideoWriter(savename, 'Uncompressed AVI');
        outputVideo = VideoWriter(savename, 'Motion JPEG AVI');
    case '.mp4'
        outputVideo = VideoWriter(savename, 'MPEG-4');
    case '.mov'
        outputVideo = VideoWriter(savename, 'Motion JPEG 2000');
    otherwise
        error('Unsupported format. Please choose avi, mp4, or mov.');
end
outputVideo.FrameRate = frameRate;

%% open videofile and save frame by frame
open(outputVideo);
numFrames = size(inputvideo, 4);
for k = 1:numFrames
    % Extract the k-th frame (now guaranteed to be RGB and uint8)
    frame = inputvideo(:,:,:,k);
    
    % Write the frame to the video file
    writeVideo(outputVideo, frame);
end

%% clode videofile
close(outputVideo);
end
