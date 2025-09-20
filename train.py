import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

COEF = 1000 # for precision
OMEGA = 1 / 2000 # for QFLIM phasor plots. 2000 ps will be located at (0.5,0.5)

def raw2features(
    frameList,                 # raw data frames
    feature,                    # lifetime/intensity/cosine/sine
):
    if len(frameList) == 0:
        return torch.tensor([], dtype=torch.float32)
    if feature == 'lifetime':
        return frameList
    if feature == 'intensity':
        return torch.stack(
            [COEF * (f>1e-10).float().sum(axis=1) for f in frameList], 
            dim=1
            )
    if feature == 'cosine':
        return [COEF * torch.cos(OMEGA * f) * (f > 1e-10) for f in frameList]
    if feature == 'sine':
        return [COEF * torch.sin(OMEGA * f) * (f > 1e-10) for f in frameList]
    
def output2save(
    output,                     # neural networ output
    feature,                    # lifetime/intensity/cosine/sine 
):
    if feature == 'lifetime':
        return output
    if feature == 'intensity':
        return output / COEF
    if feature == 'cosine':
        return output / COEF
    if feature == 'sine':
        return output / COEF
    
def computeLoss(
    output,                     # the output of the network, (batch, 1, width, height)
    input,                      # the input of the network, (batch, maxPhotonNum, width, height), maybe a list
    feature,                    # lifetime/intensity/cosine/sine 
    loss_type = 'L2',
):
    if isinstance(input, list):
        input = input[0]

    if feature in ['lifetime', 'cosine', 'sine']:
        '''
        Loss for QFLIM

        Each photon is weighted equally to compute loss

        INPUT:
            output: the output of the network, (batch, 1, width, height)
            input: the input of the network, (batch, maxPhotonNum, width, height), maybe a list

        ELiiiiiii, 20250123
        '''
        _, maxPhotonNum, _, _ = input.shape

        output_repmat = output.expand(-1, maxPhotonNum, -1, -1)

        if loss_type == 'L1':
            loss = torch.abs(output_repmat - input)
        elif loss_type == 'L2':
            loss = (output_repmat - input) ** 2
        else:
            raise ValueError("loss_type must be 'L1' or 'L2'")
        
        mask = (torch.abs(input) > 1e-10).float()
        sum_loss = (loss * mask).sum() if mask.sum() > 0 else (loss * 0.0).sum()

        return sum_loss
    
    if feature == 'intensity':
        '''
        Loss function in SUPPORT
            https://github.com/NICALab/SUPPORT/blob/main/src/train.py
            Eom, M. et al. Statistically unbiased prediction enables accurate denoising of voltage imaging data. Nature Methods (2023).

        INPUT:
            output: the output of the network, (batch, 1, width, height)
            input: the input of the network, (batch, 1, width, height)
        '''
        if loss_type == 'L1':
            return torch.nn.functional.l1_loss(output, input)
        elif loss_type == 'L2':
            return torch.nn.functional.mse_loss(output, input)
        else:
            raise ValueError("loss_type must be 'L1' or 'L2'")
        
def train(
    dataset,                   # raw data
    model,                     # model
    args,                      # parameters
    *,
    feature = 'lifetime',      # lifetime/intensity/cosine/sine
    rng = None,
    savepath = None,
    flag_testEachEpoch = False,
    flag_testLastEpoch = True,
):
    # for reproduce
    if rng is None:
        random_seed = 0 # int(time.time()) # fix it to reproduce
        rng = np.random.default_rng(random_seed)
    
    # savepath
    if savepath is None:
        savepath = args.savepath

    # generate patches
    dataset.patch_interval = copy.copy(args.patch_interval)
    dataset.gen_patch_index()

    # cuda
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
        print('\033[1;31mUsing {} GPU(s) -----> \033[0m'.format(torch.cuda.device_count()))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataloader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # epoch by epoch
    for epoch in range(0, args.n_epochs):
        model.train()

        with tqdm(total=len(loader), desc=f'Training ' + feature + f' {epoch+1}/{args.n_epochs}', unit='batch', dynamic_ncols=True) as pbar:
            for iteration, (frames_surroundings, frame_target, patch_index_stich_patch, patch_indices_stich_stack) in enumerate(loader):

                frames_surroundings, frame_target = random_transform_list(frames_surroundings, frame_target, rng)

                frames_surroundings = listToCuda(frames_surroundings)
                frame_target = listToCuda(frame_target)

                input_surroundings = raw2features(frames_surroundings, feature)
                input_target = raw2features(frame_target, feature)

                optimizer.zero_grad()

                output = model(input_surroundings, input_target)
                loss_l1 = computeLoss(output, input_target, feature, loss_type='L1') * 0.0
                loss_l2 = computeLoss(output, input_target, feature, loss_type='L2') * 1.0

                loss_sum = loss_l1 + loss_l2
                loss_sum.backward()
                optimizer.step()

                pbar.set_postfix_str(f'Loss: {loss_sum.item():.2f}')
                pbar.update(1)

            torch.save(model.state_dict(), f"{savepath}//model_{feature}_{epoch}.pth")
            torch.save(optimizer.state_dict(), f"{savepath}//optimizer_{feature}_{epoch}.pth")

        # test
        do_test = False
        if flag_testEachEpoch:
            do_test = True
        elif flag_testLastEpoch and epoch == args.n_epochs - 1:
            do_test = True

        if do_test:
            dataset.patch_interval[0] = 1
            dataset.gen_patch_index()

            # dataloader
            loaderTest = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            with torch.no_grad():
                model.eval()

                # initialize
                output_stack = []
                for acqCount in range(len(dataset.acqs)):
                    output_stack.append(np.zeros(tuple(dataset.getShape(acqCount)), dtype=np.float32))

                # stitch stack
                for _, (frames_surroundings, frame_target, patch_index_stich_patch, patch_indices_stich_stack) in enumerate(tqdm(loaderTest, desc="validate")):
                    
                    frames_surroundings = listToCuda(frames_surroundings)
                    frame_target = listToCuda(frame_target)

                    input_surroundings = raw2features(frames_surroundings, feature)
                    input_target = raw2features(frame_target, feature)

                    optimizer.zero_grad()

                    output = model(input_surroundings, input_target)

                    for bi in range(output.shape[0]):
                        acq_idx, t_target, patch_x_start, patch_x_end, patch_y_start, patch_y_end = patch_index_stich_patch[bi].int().tolist()
                        acq_idx, t_target, stack_x_start, stack_x_end, stack_y_start, stack_y_end = patch_indices_stich_stack[bi].int().tolist()

                        output_stack[acq_idx][t_target, stack_x_start:stack_x_end, stack_y_start:stack_y_end] \
                            = output2save(output[bi].squeeze()[patch_x_start:patch_x_end, patch_y_start:patch_y_end].cpu().numpy(), feature)
                        
                for idx, output_stack_here in enumerate(output_stack):
                    saveastiff(output_stack_here, f"{savepath}//output_{feature}_{idx}_epoch{epoch}.tif")

            dataset.patch_interval = copy.copy(args.patch_interval)
            dataset.gen_patch_index()

    # return
    return output_stack