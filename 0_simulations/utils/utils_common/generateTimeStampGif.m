% Generate time-stamp gif and save it to savepath

%%ELiiiiiii, 20211112
function generateTimeStampGif(intervelInSeconds,frameNumber,totalDuration,savepath,startTimeInSeconds,dispformat,endStr,color)
if nargin < 3
    error('Not enough inputs');
end
if nargin < 4
    savepath = '.';
end
if nargin < 5 || isempty(startTimeInSeconds)
    startTimeInSeconds = 0;
end
if nargin < 6
    dispformat ='HH:MM:SS';% 'YYYY:mm:DD:HH:MM:SS.FFF';%
end
if nargin < 7
    endStr = '';
end
if nargin < 8
    color = 'k';
end

%%
fileName = [savepath, '//timeStamp_', datestr(now, 'YYYYmmDD_HHMMSS'), '.gif'];
warning off;
%% start time
%default start time: 2000:01:01:00:00:00
starttime = datenum(2000, 1, 1, 0, 0, 0) + startTimeInSeconds/86400;

timeNow = starttime;
for frameCount = 1:frameNumber
    dispstr = [datestr(timeNow, dispformat), endStr];
%     disp(dispstr);
    [I,A, ~, ~] = str2im(dispstr,'Color',color);
    I = I(5:end,:,:);
    [SIf,cm] = rgb2ind(I,256);
    if frameCount == 1
        imwrite(SIf,cm,fileName,'gif','Loopcount',inf,'DelayTime',totalDuration/frameNumber);
    else
        imwrite(SIf,cm,fileName,'gif','WriteMode','append','DelayTime',totalDuration/frameNumber);
    end
%     imwrite(I,['I',num2str(frameCount),'.png'],'Al',A);%

    timeNow = timeNow + intervelInSeconds/86400;
end
warning on;