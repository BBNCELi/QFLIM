import argparse
import os
from dataset import FLIMDataset
from model import SUPPORT, QFLIM
from train import train
from utils import findAndLoadMat, inwlt, saveastiff_RGB

def main():
    #######################################################################################
    ### Parameters ###
    #######################################################################################
    parser = argparse.ArgumentParser(description="Running QFLIM.")

    ## input data and savepath
    parser.add_argument("--folderName", type=str, default='.//simu_USAF1951_PPP0.5//raw', help="folder of raw photons")
    parser.add_argument("--savepath", type=str, default=None, help="save results here")

    ## training parameters
    parser.add_argument("--gpu", type=str, default='0', help="GPU id to use")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[61, 128, 128], metavar=("D", "H", "W"), help="patch size as three ints: D H W (default: 61 128 128)")
    parser.add_argument("--patch_interval", type=int, nargs=3, default=[10, 64, 64], metavar=("D", "H", "W"), help="patch interval as three ints: D H W (default: 10 64 64)")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--flag_testEachEpoch", type=bool, default=False, help="test each epoch or not")
    parser.add_argument("--flag_testLastEpoch", type=bool, default=True, help="test last epoch or not")

    ## Network parameters for both lifetime and intensity denoising.
    ## We adopt the network architecture from SUPPORT, which gives extraordinary performance
    ## by makeing full use of spatial-temporal information.
    ## Eom, M. et al. Statistically unbiased prediction enables accurate denoising of voltage imaging data. Nature Methods (2023).
    ## https://github.com/NICALab/SUPPORT
    parser.add_argument("--depth", type=int, default=5, help="the number of blind spot convolutions, must be an odd number")
    parser.add_argument("--blind_conv_channels", type=int, default=64, help="the number of channels of blind spot convolutions")
    parser.add_argument("--one_by_one_channels", type=int, default=[32, 16], nargs="+", help="the number of channels of 1x1 convolutions")
    parser.add_argument("--last_layer_channels", type=int, default=[64, 32, 16], nargs="+", help="the number of channels of 1x1 convs after UNet")
    parser.add_argument("--bs_size", type=int, default=[3, 3], nargs="+", help="the size of the blind spot")
    parser.add_argument("--bp", action="store_true", help="blind plane")
    parser.add_argument("--unet_channels", type=int, default=[16, 32, 64, 128, 256], nargs="+", help="the number of channels of UNet")
    args = parser.parse_args()

    if not os.path.exists(args.folderName):
        raise ValueError("Please give a valid folder that contains each frame as a single file")
    if not os.path.isdir(args.folderName):
        raise ValueError("Please give a folder that contains each frame as single file")
    if not any(f.lower().endswith(".tif", ".tiff") for f in os.listdir(args.folderName)):
        raise ValueError("Please give a valid folder that contains each frame as a single .tif file")
    if args.savepath is None:
        args.savepath =  os.path.dirname(args.folderName) + f'//QFLIM'
    os.makedirs(args.savepath, exist_ok=True)
    args.ngpu = str(args.gpu).count(',') + 1
    args.input_frames = args.patch_size[0]
    args.batch_size = 1

    #######################################################################################
    ### Dataset ###
    #######################################################################################
    datasethere = FLIMDataset(args.patch_size, args.patch_interval)
    datasethere.addFrames_tiff(folderName = args.folderName)

    #######################################################################################
    ### Lifetime denoising ###
    #######################################################################################
    model_lifetime = QFLIM(in_channels=args.input_frames, mid_channels=args.unet_channels, depth=args.depth,\
            blind_conv_channels=args.blind_conv_channels, one_by_one_channels=args.one_by_one_channels,\
                    last_layer_channels=args.last_layer_channels, bs_size=args.bs_size, bp=args.bp)
    lifetime_guess = train(datasethere, model_lifetime, args,
                            feature = 'lifetime',
                            flag_testEachEpoch = args.flag_testEachEpoch,
                            flag_testLastEpoch = args.flag_testLastEpoch
                            )

    #######################################################################################
    ### Intensity denoising ###
    #######################################################################################
    # We use SUPPORT for intensity denoising, because of its extraordinary performance in photon-starved conditions.
    model_intensity = SUPPORT(in_channels=args.input_frames, mid_channels=args.unet_channels, depth=args.depth,\
        blind_conv_channels=args.blind_conv_channels, one_by_one_channels=args.one_by_one_channels,\
                last_layer_channels=args.last_layer_channels, bs_size=args.bs_size, bp=args.bp)
    intensity_guess = train(datasethere, model_intensity, args,
                            feature = 'intensity',
                            flag_testEachEpoch = args.flag_testEachEpoch,
                            flag_testLastEpoch = args.flag_testLastEpoch
                            )

    #######################################################################################
    ### RGB visualization ###
    #######################################################################################
    lt_contrast = [500, 3500]
    in_contrast = [0, 1]
    colormapName = 'weddingdayblues'
    flag_white = 0
    cm  = findAndLoadMat(colormapName+f"_double.mat", "..//..//")

    results_lt_QFLIM_RGB = inwlt(lifetime_guess[0], intensity_guess[0], lt_contrast, in_contrast, cm, flag_white)
    dropFrames = int((args.patch_size[0] - 1) / 2)
    results_lt_QFLIM_RGB = results_lt_QFLIM_RGB[dropFrames:-dropFrames,:,:,:]

    lt_low, lt_high = lt_contrast
    in_low, in_high = in_contrast
    savename = args.savepath + f"//output_QFLIM_lt{lt_low}-{lt_high}_in{in_low}-{in_high}_{colormapName}.tif"
    saveastiff_RGB(results_lt_QFLIM_RGB, savename)

if __name__ == "__main__":
    main()