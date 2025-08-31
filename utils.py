import tifffile
import torch
import numpy as np
import os
from scipy.io import loadmat

def loadtiff(imgPath):
    npimg = tifffile.imread(imgPath)
    return npimg

def saveastiff(npimg, name):
    if isinstance(npimg, torch.Tensor):
        npimg = npimg.detach().cpu().numpy()
    # overwrite existed file
    # tifffile.imsave(name, npimg)
    tifffile.imwrite(name, npimg)
    return

def saveastiff_RGB(arr_rgb, name):
    arr_rgb = np.asarray(arr_rgb, dtype=np.uint8)
    tifffile.imwrite(name, arr_rgb, photometric='rgb')
    return

def loadFirstVariableFromMat(path):
    data = loadmat(path)
    for key, value in data.items():
        if not key.startswith("__"):
            return value
    raise ValueError(f"No valid variable found in {path}")

def findAndLoadMat(nameMat, root_dir):
    pathMat = None
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == nameMat:
                pathMat = os.path.join(dirpath, f)

    if pathMat is not None:
        return loadFirstVariableFromMat(pathMat)
    else:
        return None

def move_zeros_to_end_3d(input_tensor):
    """
    Fully vectorized version.
    Moves non-zero values to the front along Z for each pixel, pushes zeros to the back.
    Truncates to the max non-zero depth.
    Input: [Z, H, W]
    Output: [Z_new, H, W]
    """
    Z, H, W = input_tensor.shape
    input_flat = input_tensor.view(Z, -1)  # [Z, H*W]

    # Create mask of non-zeros
    non_zero_mask = input_flat > 1e-7  # [Z, H*W]
    non_zero_counts = non_zero_mask.sum(dim=0)  # [H*W]

    # Create sorting keys: non-zeros (True=1) should come before zeros (False=0)
    sort_keys = non_zero_mask.to(torch.int32)  # [Z, H*W]

    # Sort so that non-zeros come first
    sorted_vals, _ = torch.sort(input_flat, dim=0, descending=True)
    sorted_vals = sorted_vals.view(Z, H, W)

    # Truncate to max depth with non-zero
    max_depth = int(non_zero_counts.max().item())
    output_truncate = sorted_vals[:max_depth, :, :]
    max_nonzero_depth = non_zero_counts.view(H, W)

    return sorted_vals, output_truncate, max_nonzero_depth

def nanmean_torch(input, dim):
    '''
    A manual implementation of torch.nanmean for pytorch versions before 2.5

    ELiiiiiii, 20250122
    '''

    # replace nans
    input_nonan = torch.where(torch.isnan(input), torch.tensor(0.0, device=input.device), input)
    
    # Count non-NaN elements
    count = torch.sum(~torch.isnan(input), dim=dim)
    
    # sum and divide by count
    mean_nonan = torch.sum(input_nonan, dim=dim) / count

    # for where count == 0
    return torch.where(count == 0, torch.tensor(0.0, device=input.device), mean_nonan)

def mean_nz(input, axis=0):
    '''
    The average of non-zero elements along given axis, accepts both numpy.ndarray && torch.tensor

    ELiiiiiii, 20250122
    '''

    # for numpy.ndarray
    if isinstance(input, np.ndarray):
        if not np.issubdtype(input.dtype, np.floating):
            input = input.astype(float)
        input = input.copy()
        input[input < 1e-10] = np.nan
        with np.errstate(invalid='ignore'):
            input_meannz = np.nanmean(input, axis=axis)
        input_meannz = np.nan_to_num(input_meannz, nan=0.0)
        return input_meannz

    # for torch.tensor
    elif isinstance(input, torch.Tensor):
        if not input.is_floating_point():
            input = input.float()
        input = input.clone()
        input[input < 1e-10] = float('nan')
        input_meannz = nanmean_torch(input, dim=axis)
        input_meannz = torch.where(torch.isnan(input_meannz), 
                                   torch.tensor(0.0, dtype=input_meannz.dtype, device=input_meannz.device), 
                                   input_meannz
                                   )
        return input_meannz

    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

def listToCuda(list, cuda_device=0):
    '''
    Put a list of tensor to cuda

    ELiiiiiii, 20250123
    '''

    result = []
    for item in list:
        result.append(item.cuda(cuda_device))

    return result

def random_transform_list(frames_surroundings, frame_target, rng):
    '''
    Transform a list of frames randomly

    INPUT:
    frames_surroundings: a list of FLIM frames
                         each frame is a pytorch tensor with dimension [b, maxPhotonNum, width, height]
    frame_target: a list of target FLIM frame
                  each frame is a pytorch tensor with dimension [b, maxPhotonNum, width, height]

    Derived from the SUPPORT version 'random_transform'
    https://github.com/NICALab/SUPPORT/blob/main/src/utils/dataset.py
    ELiiiiiii, 20250123
    '''

    if rng is None:
        rng = np.random.default_rng()

    rn_r = rng.integers(0, 4) # random number for rotation
    rn_f = rng.integers(0, 2) # random number for flip

    # rotate
    if rn_r == 1:
        frames_surroundings = [torch.rot90(f, k=1, dims=(2,3)) for f in frames_surroundings]
        if frame_target is not None:
            frame_target = [torch.rot90(f, k=1, dims=(2,3)) for f in frame_target]
    elif rn_r == 2:
        frames_surroundings = [torch.rot90(f, k=2, dims=(2,3)) for f in frames_surroundings]
        if frame_target is not None:
            frame_target = [torch.rot90(f, k=2, dims=(2,3)) for f in frame_target]
    elif rn_r == 3:
        frames_surroundings = [torch.rot90(f, k=3, dims=(2,3)) for f in frames_surroundings]
        if frame_target is not None:
            frame_target = [torch.rot90(f, k=3, dims=(2,3)) for f in frame_target]
    
    if rn_f == 1:
        frames_surroundings = frames_surroundings[::-1]
        frame_target = frame_target[::-1]

    return frames_surroundings, frame_target

def random_transform(input, target, rng, is_rotate=True):
    """
    SUPPORT version of random_transform
    https://github.com/NICALab/SUPPORT/blob/main/src/utils/dataset.py

    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4) # random number for rotation
    rand_num_2 = rng.integers(0, 2) # random number for flip 

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))
    
    if rand_num_2 == 1:
        input = torch.flip(input, dims=[1])
        if target is not None:
            target = torch.flip(target, dims=[1])

    return input, target

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def inwlt(
    lt_in,
    in_in=None,
    lt_contrast=None,
    in_contrast=None,
    colormap="jet",
    flag_white=False,
):
    lt_in = to_numpy(lt_in)
    in_in = to_numpy(in_in)

    lt = np.asarray(lt_in, dtype=np.float64)
    inten = lt.copy() if in_in is None else np.asarray(in_in, dtype=np.float64)

    if lt.shape != inten.shape:
        raise ValueError("Size mismatch: input intensity and lifetime must have the same shape.")

    # colormap
    if isinstance(colormap, str):
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap, 256)(np.linspace(0, 1, 256))[:, :3]
    else:
        cmap = np.asarray(colormap, dtype=np.float64)
        if cmap.shape != (256, 3) or (cmap.min() < 0) or (cmap.max() > 1):
            raise ValueError("colormap must be 256x3 in [0,1].")

    if lt_contrast is None:
        lt_contrast = (0.0, float(np.nanmax(lt)) if np.isfinite(np.nanmax(lt)) else 1.0)
    if in_contrast is None:
        in_contrast = (0.0, float(np.nanmax(inten)) if np.isfinite(np.nanmax(inten)) else 1.0)

    def _rescale(x, a, b):
        if b <= a:
            y = np.zeros_like(x, dtype=np.float64)
        else:
            y = (x - a) / (b - a)
            y = np.clip(y, 0.0, 1.0)
        return y

    lt_res = _rescale(lt, lt_contrast[0], lt_contrast[1])
    in_res = _rescale(inten, in_contrast[0], in_contrast[1])

    lt_idx = (lt_res * 255.0).astype(np.uint8)
    colors = cmap[lt_idx.reshape(-1)]  # (N,3)
    in_flat = in_res.reshape(-1)

    if not flag_white:
        rgb = 255.0 * (colors * in_flat[:, None])
    else:
        rgb = 255.0 * (in_flat[:, None]*colors + (1.0 - in_flat)[:, None]*1.0)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb.reshape((*lt.shape, 3))

# tests
if __name__ == "__main__":
    #### test 1
    from copy import deepcopy
    input_tensor = torch.tensor([
        [[1.0, 0.0],
         [0.0, 3.0]],
        [[0.0, 4.0],
         [0.0, 0.0]],
        [[2.0, 0.0],
         [5.0, 0.0]],
        [[0.0, 0.0],
         [0.0, 0.0]],
        [[6.0, 0.0],
         [7.0, 0.0]],
    ])  # Shape: [5, 2, 2]

    original_input = deepcopy(input_tensor)

    # Call the function
    full_sorted, truncated, max_depths = move_zeros_to_end_3d(input_tensor)

    # Check 1: sorted result has non-zeros at the front
    for h in range(input_tensor.shape[1]):
        for w in range(input_tensor.shape[2]):
            sorted_stack = full_sorted[:, h, w]
            nonzeros = sorted_stack[sorted_stack > 1e-7]
            zeros = sorted_stack[sorted_stack <= 1e-7]
            assert torch.all(nonzeros >= 1e-7), f"Non-zeros check failed at ({h},{w})"
            assert torch.all(zeros <= 1e-7), f"Zeros check failed at ({h},{w})"
            assert torch.allclose(sorted_stack, torch.cat((nonzeros, zeros))), f"Sorting failed at ({h},{w})"

    # Check 2: truncated shape matches max non-zero depth
    expected_max_depth = int(max_depths.max().item())
    assert truncated.shape[0] == expected_max_depth, "Truncated depth mismatch"

    # Check 3: Input is unchanged
    assert torch.allclose(input_tensor, original_input), "Input tensor was modified"

    print("✅ All tests passed!")



    #### test 2
    np_input = np.array([[[0, 1, 0], [2, 0, 3]], [[4, 5, 0], [0, 6, 7]], [[8, 9, 0], [0, 0, 0]]])
    torch_input = torch.tensor(np_input)

    # Test randChooseOne_nz
    np_output1 = randChooseOne_nz(np_input, axis=0)
    torch_output1 = randChooseOne_nz(torch_input, axis=0)

    # Test randChooseTwo_nz
    np_output2, np_output2_2 = randChooseTwo_nz(np_input, axis=0)
    torch_output2, torch_output2_2 = randChooseTwo_nz(torch_input, axis=0)

    print('numpy tests:')
    print('input:')
    print(np_input)
    print('randChooseOne_nz:')
    print(np_output1)
    print('randChooseTwo_nz_1:')
    print(np_output2)
    print('randChooseTwo_nz_2:')
    print(np_output2_2)

    print('torch tests:')
    print('input:')
    print(torch_input)
    print('randChooseOne_nz:')
    print(torch_output1)
    print('randChooseTwo_nz_1:')
    print(torch_output2)
    print('randChooseTwo_nz_2:')
    print(torch_output2_2)