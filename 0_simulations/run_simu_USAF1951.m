% This script gives a comprehensive simulation of photon detection in FLIM.
% 
% In FLIM, both photon counts and arrival times are affected by noise. 
% The noises can grouped into the following categories:

% 1. Sample photon number. For each pulse delivered to a pixel, the number
%     of photons emitted from the sample follows a Poisson distribution. 
%     In traditional TCSPC, only the first emitted photon can be detected 
%     (pile-up), and in modern optics as well as electronics this 
%     restriction is relaxed. In practice, each pixel may receive several 
%     pulses. We use PPP (the averaged number of photons per pixel coming 
%     from the sample) to denote the total photon number.
%   Paras: PPP

% 2. Sample photon arrival times. Photon arrival times follow an 
%     single-exponential distribution and may deviate in more complex cases.
%   Paras: \tao

% 3. Background ambient photon number and their arrival times. In two-photon 
%     and three-photon microscopy, background signals are substantially 
%     suppressed due to the nonlinear excitation process, and are often 
%     negligible under standard imaging conditions. When present, background 
%     photons originate from ambient environmental light. The occurrence of 
%     these photons can be modeled as a Poisson process, in which the 
%     inter-arrival times follow an exponential distribution. In practice, 
%     we use BPP (averaged number of background photons per pixel) to 
%     denote the background, simulate the photon counts using a Poisson 
%     distribution and the arrival times with uniform distribution.
%   NOTICE: in one-photon microscopy background usually means sample
%     background photons, due to the excitation of the whole volume. These
%     background photons may deviate sample photon arrival times a lot from 
%     the original sinlge-exponential distribution.
%   Paras: BPP

% 4. IRF. In practice, the detected photon arrival times are broadened by 
%     electronic noise, the finite width of excitation laser pulse, 
%     detector timing jitter, optical dispersion in the system, and others.
%     These effects are collectively described by the instrument response 
%     function (IRF). We approximate IRF using Gaussian functions with 
%     varying widths.
%   Paras: IRF_sigma

%%%ELiiiiiii, 20250825

%% preparations
addpath(genpath('./utils'));
timenow = datestr(now, 'YYYYmmDD_HHMMSS');
savefolder = ['..//simu_USAF1951_', 'PPP0.5'];
disp(['Results will be saved at ', savefolder]);
mkdir(savefolder);

%% load sample lifetime
sampleName = './/sample_lt//USAF512_500-500-3000ps.tif';
sample = single(loadtiff(sampleName)); % ps

%% set parameters
% PPP, BPP, and IRF_sigma
ppp = 0.5;
bpp = 0;
IRF_sigma = 0; % ps

% pulse interval, ps
% In practice the pulse interval should usually be >10X than mean lifetimes
pulse_interval_ps = 10 * max(sample(:));

% pile-up: in future techniques pile-up effects may be completely eliminated
% In practice pile-up can be ignored when photons per pulse (pppulse) < 0.05
% We here ignore pile-ups
pileup = false;

% system parameters
pulsesPerPixel = 1;
pppulse = ppp ./ pulsesPerPixel; % photons per pulse
pppulse = pppulse .* ones(size(sample)) .* (sample>0); % no fluorophores, no photons
bppulse = bpp ./ pulsesPerPixel;
bppulse = bppulse .* ones(size(sample)); % no fluorophores, still photons

%% initializations
frameNumber = 500;
results_lt_gt = zeros([size(sample),frameNumber], 'single');
results_lt_fastflim = zeros([size(sample),frameNumber], 'single');
results_in = zeros([size(sample),frameNumber], 'single');
%% simulate photon arrivals frame by frame
for frameCount = 1:frameNumber
    fprintf('frame %d ||| %d', frameCount, frameNumber);

    results_lt_gt(:,:,frameCount) = sample;
    results_in_detected_thisFrame = [];
    results_lt_detected_thisFrame = [];
    %%% pulse by pulse
    for pulseCount = 1:pulsesPerPixel
        fluoPhotonsEmit = poissrnd(pppulse);
        backgroundPhotonsEmit = poissrnd(bppulse);
        photonsEmit = fluoPhotonsEmit + backgroundPhotonsEmit;

        results_in_detected_thisFrame = cat(3,results_in_detected_thisFrame, photonsEmit);
        
        if any(photonsEmit, 'all')
            %%% sample photons
            fluoPhotonsEmit_max = max(fluoPhotonsEmit, [], 'all');
            if fluoPhotonsEmit_max == 0
                lt_detected_sample = zeros(size(sample), 'single');
            else
                lt_detected_sample = exprnd(repmat(sample,[1,1,fluoPhotonsEmit_max]));
                [~, ~, Z] = meshgrid(1:size(sample,2), 1:size(sample,1), 1:fluoPhotonsEmit_max);
                mask = Z <= repmat(fluoPhotonsEmit, [1, 1, fluoPhotonsEmit_max]);
                lt_detected_sample(~mask) = 0;
            end
            
            %%% background photons
            backgroundPhotonsEmit_max = max(backgroundPhotonsEmit, [], 'all');
            if backgroundPhotonsEmit_max == 0
                lt_detected_background = zeros(size(sample), 'single');
            else
                lt_detected_background = pulse_interval_ps * rand([size(sample,2),size(sample,1),backgroundPhotonsEmit_max]);
                [~, ~, Z] = meshgrid(1:size(sample,2), 1:size(sample,1), 1:backgroundPhotonsEmit_max);
                mask = Z <= repmat(backgroundPhotonsEmit, [1, 1, backgroundPhotonsEmit_max]);
                lt_detected_background(~mask) = 0;
            end

            %%% pile-up
            lt_detected_thisPulse = cat(3,lt_detected_sample, lt_detected_background);
            if pileup
                lt_detected_thisPulse = min_nz(lt_detected_thisPulse, 3);
            end

            %%%IRF
            noise = (randn(size(sample)) * IRF_sigma ) .* (lt_detected_thisPulse ~= 0);
            lt_detected_thisPulse = lt_detected_thisPulse + noise;
            % find those photons with lifetime < 0
            m = (lt_detected_thisPulse<0);
            lt_detected_thisPulse(m) = lt_detected_thisPulse(m) + pulse_interval_ps;

        else
            lt_detected_thisPulse = zeros(size(sample), 'single');
        end

        results_lt_detected_thisFrame = cat(3, results_lt_detected_thisFrame, lt_detected_thisPulse);
    end

    %%% saveEachFrame
    fprintf(['... Photon counts: ', num2str(sum(results_in_detected_thisFrame(:))), '\n']);
%     [~, results_lt_detected_thisFrame] = moveZerosToEnd_3d(results_lt_detected_thisFrame); % to delete 0s
    opt.message = false;
    saveastiff_overwrite(results_lt_detected_thisFrame, [savefolder, '//raw//frame', num2str(frameCount, '%.5d'), '.tif'], opt);

    results_lt_fastflim(:,:,frameCount) = mean_nz(results_lt_detected_thisFrame, 3);
    results_in(:,:,frameCount) = sum(results_in_detected_thisFrame, 3);
end

saveastiff_overwrite(results_lt_gt, [savefolder, '//lt_gt//lt_gt.tif']);
saveastiff_overwrite(results_lt_fastflim, [savefolder, '//lt_fastflim//lt_fastflim.tif']);

%% save RGB -- intensity weighted lifetime images
lt_contrast = [500, 3500];
in_contrast = [0, ppp];
colormapName = 'weddingdayblues';
flag_white = 0;
cm  = loadFirstVariable([colormapName, '_double.mat']);

%%% save lifetime gt RGB
results_lt_gt_RGB = inwlt(results_lt_gt, ppp .* (results_lt_gt>0), lt_contrast, in_contrast, cm, flag_white);
saveastiff_RGBAsLastDim(results_lt_gt_RGB,...
    [savefolder, '//lt_gt//lt_gt',...
    '_lt', array2str(lt_contrast, '-'),...
    '_in', array2str(in_contrast,'-'),...
    '_', colormapName, '.tif']);
%%% save fastflim RGB
results_lt_fastflim_RGB = inwlt(results_lt_fastflim, results_in, lt_contrast, in_contrast, cm, flag_white);
saveastiff_RGBAsLastDim(results_lt_fastflim_RGB,...
    [savefolder, '//lt_fastflim//lt_fastflim',...
    '_lt', array2str(lt_contrast, '-'),...
    '_in', array2str(in_contrast,'-'),...
    '_', colormapName, '.tif']);
