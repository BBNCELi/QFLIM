import tifffile
import os
import torch
import numpy as np
import math
from utils import *


def get_coordinate(img_size, patch_size, patch_interval):
    """DeepCAD version of stitching
    https://github.com/cabooster/DeepCAD/blob/53a9b8491170e298aa7740a4656b4f679ded6f41/DeepCAD_pytorch/data_process.py#L374
    """
    whole_s, whole_h, whole_w = img_size
    img_s, img_h, img_w = patch_size
    gap_s, gap_h, gap_w = patch_interval

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    # print(whole_s, whole_h, whole_w)
    # print(img_s, img_h, img_w)
    # print(gap_s, gap_h, gap_w)

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)

    coordinate_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    if num_w > 1:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    else:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    if num_h > 1:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    else:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = x*gap_h+img_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    if num_s > 1:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s-cut_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                coordinate_list.append(single_coordinate)

    return coordinate_list


class FLIMDataset:
    '''
    A class to load and organize data for training and testing CYNTAFLIM

    PLEASE pay special attention to the FLIM data structure

    The input data will be structed into a two-layer hierarchy:

    - FLIMDataset: the entire dataset
        - acqs: a list of different time-series acquisitions (e.g. sessions or recordings or z-slices)
            - frames: a list of time-series frames within each acquisition
                - arriveTimes: maxPhotonNum * width * height
                                each pixel contains the arrive time of a photon
                                0 if no photon
        - acqs_fastFLIM: a list of fastFLIM videos of each acquisition
            - frames_fastFLIM: frameNum * width * height
                                each pixel contains the averaged photon arrive time
                                0 if no photon
        - acqs_intensity: a list of intensity videos of each acquisition
            - frames_intensity: frameNum * width * height
                                each pixel contains the number of photons arrived

    '''

    #######################################################################################
    ### Initializations and property functions ###
    #######################################################################################
    def __init__(self, patch_size = [61,128,128], patch_interval=[1, 64, 64]):
        print(f"Constructing a FLIMDataset")
        self.dataType = torch.float32
        self.acqs = []
        self.acqs_fastFLIM = []
        self.acqs_intensity = []

        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")      
        
        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.patch_indices = []
        self.patch_indices_stich_patch = []
        self.patch_indices_stich_stack = []


    def __len__(self):
        return len(self.patch_indices)
    
    # the shape of the ith acquisition
    def getShape(self, i=-1):
        if i >= len(self.acqs):
            print(f"Error: the {i}th acquisition can not be accessed")
            return None
        
        elif i < 0:
            resultShape = []
            for acq in self.acqs:
                frameNum = len(acq)
                height, width = acq[0].shape[-2:]
                resultShape.append(torch.Size([frameNum, height, width]))
            return resultShape
        
        else:
            frameNum = len(self.acqs[i])
            height, width = self.acqs[i][0].shape[-2:]
            return torch.Size([frameNum, height, width])


    #######################################################################################
    ### Add new acquisitions ###
    #######################################################################################
    # Add a tiff file which contains multiple frames
    def addAcq_tiff(self, fileName_fastFLIM, filename_intensity=None):
        if not fileName_fastFLIM.endswith((".tif", ".tiff")):
            print(f"Error: {fileName_fastFLIM} is not a tiff file")
            return
        print(f"Loading {fileName_fastFLIM}...")

        frames = torch.from_numpy(tifffile.imread(fileName_fastFLIM)).to(self.dataType)

        if frames.ndim < 3:
            frames = frames.unsqueeze(0)
        acq = [frame.unsqueeze(0) for frame in frames]
        acq_fastFLIM = frames
        if filename_intensity is not None:
            print(f"Loading {filename_intensity}...")
            acq_intensity = torch.from_numpy(tifffile.imread(filename_intensity)).to(self.dataType)
            assert acq_fastFLIM.shape == acq_intensity.shape , "The intensity video should have the exact same size with the fastFLIM video"
        else:
            acq_intensity = (acq_fastFLIM > 0).to(self.dataType)

        self.acqs.append(acq)
        self.acqs_fastFLIM.append(acq_fastFLIM)
        self.acqs_intensity.append(acq_intensity)
    
    # Add multiple frames from a folder
    def addFrames_tiff(self, folderName, fileIndStart=0, fileIndEnd=None, groupAvg=1):
        if not os.path.isdir(folderName):
            print(f"Error: {folderName} is not a valid directory")
            return
        print(f"Loading TIFF files from {folderName}")

        frameFileNamesToBeLoaded = []
        for item in os.listdir(folderName):
            if item.endswith((".tif", ".tiff")):
                frameFileNamesToBeLoaded.append(item)
        
        frameFileNamesToBeLoaded = sorted(frameFileNamesToBeLoaded)

        if fileIndEnd is None:
            fileIndEnd = len(frameFileNamesToBeLoaded)

        frameFileNamesToBeLoaded = frameFileNamesToBeLoaded[fileIndStart:fileIndEnd]

        acq = []
        acq_fastFLIM = []
        acq_intensity = []
        currentGroup = []
        for filename in frameFileNamesToBeLoaded:
            print(f"Loading {filename}")
            thisFrame = torch.from_numpy(tifffile.imread(os.path.join(folderName, filename))).to(self.dataType)
            if thisFrame.ndim < 3:
                thisFrame = thisFrame.unsqueeze(0)

            currentGroup.append(thisFrame)

            if len(currentGroup) == groupAvg:
                catFrames = torch.cat(currentGroup, dim=0)
                _, catFrames, _ = move_zeros_to_end_3d(catFrames)
                acq.append(catFrames)
                acq_fastFLIM.append(mean_nz(catFrames, axis=0))
                acq_intensity.append(torch.sum(catFrames > 0, dim=0).to(self.dataType))
                currentGroup = []

        self.acqs.append(acq)
        self.acqs_fastFLIM.append(torch.stack(acq_fastFLIM))
        self.acqs_intensity.append(torch.stack(acq_intensity))

    ## Add frames of a video from multiple folders
    def addVideoFrames_tiff(self, folderNameList, groupAvg=1):
        frameFileNamesToBeLoaded = []
        for folderName in folderNameList:
            if not os.path.isdir(folderName):
                print(f"Error: {folderName} is not a valid directory")
                return
            
            # print(f"Loading TIFF files from {folderName}")

            frameFileNamesInThisFolder = []
            for item in os.listdir(folderName):
                if item.endswith((".tif", ".tiff")):
                    frameFileNamesInThisFolder.append(item)
        
            frameFileNamesInThisFolder = sorted(frameFileNamesInThisFolder)

            for f in frameFileNamesInThisFolder:
                frameFileNamesToBeLoaded.append(os.path.join(folderName, f))

        acq = []
        acq_fastFLIM = []
        acq_intensity = []
        currentGroup = []
        for filename in frameFileNamesToBeLoaded:
            print(f"Loading {filename}")
            thisFrame = torch.from_numpy(tifffile.imread(filename)).to(self.dataType)
            if thisFrame.ndim < 3:
                thisFrame = thisFrame.unsqueeze(0)

            currentGroup.append(thisFrame)

            if len(currentGroup) == groupAvg:
                catFrames = torch.cat(currentGroup, dim=0)
                _, catFrames, _ = move_zeros_to_end_3d(catFrames)
                acq.append(catFrames)
                acq_fastFLIM.append(mean_nz(catFrames, axis=0))
                acq_intensity.append(torch.sum(catFrames > 0, dim=0).to(self.dataType))
                currentGroup = []

        self.acqs.append(acq)
        self.acqs_fastFLIM.append(torch.stack(acq_fastFLIM))
        self.acqs_intensity.append(torch.stack(acq_intensity))

    #######################################################################################
    ### Patches ###
    #######################################################################################
    # generate patch indices
    def gen_patch_index(self):
        # print(f"Generating patch indices for the dataset...")

        patch_num = 0
        patch_indices = []
        patch_indices_stich_patch = []
        patch_indices_stich_stack = []
        for acq_idx, _ in enumerate(self.acqs):
            coordinate_list = get_coordinate(self.getShape(acq_idx), self.patch_size, self.patch_interval)
            for coordinate in coordinate_list:
                t_start = coordinate['init_s']
                t_end = coordinate['end_s']
                x_start = coordinate['init_h']
                x_end = coordinate['end_h']
                y_start = coordinate['init_w']
                y_end = coordinate['end_w']
                patch_indices.append((acq_idx, t_start, t_end, x_start, x_end, y_start, y_end))
                patch_num += 1

                if np.all(self.patch_size >= self.patch_interval):
                    t_target = 1
                    x_start = coordinate['patch_start_h']
                    x_end = coordinate['patch_end_h']
                    y_start = coordinate['patch_start_w']
                    y_end = coordinate['patch_end_w']
                    patch_indices_stich_patch.append((acq_idx, t_target, x_start, x_end, y_start, y_end))
                    t_target = int((coordinate['end_s']+coordinate['init_s']) / 2)
                    x_start = coordinate['stack_start_h']
                    x_end = coordinate['stack_end_h']
                    y_start = coordinate['stack_start_w']
                    y_end = coordinate['stack_end_w']
                    patch_indices_stich_stack.append((acq_idx, t_target, x_start, x_end, y_start, y_end))
                else:
                    patch_indices_stich_patch.append((acq_idx, 0.0, 0.0, 0.0, 0.0, 0.0))
                    patch_indices_stich_stack.append((acq_idx, 0.0, 0.0, 0.0, 0.0, 0.0))

        self.patch_indices = patch_indices
        self.patch_indices_stich_patch = patch_indices_stich_patch
        self.patch_indices_stich_stack = patch_indices_stich_stack
        # print(f"{patch_num} patches in total generated")


    #######################################################################################
    ### Get item ###
    #######################################################################################
    def __getitem__(self, i):
        patch_index = self.patch_indices[i]
        acq_idx, t_start, t_end, x_start, x_end, y_start, y_end = patch_index
        patch_index_stich_patch = self.patch_indices_stich_patch[i]
        patch_indices_stich_stack = self.patch_indices_stich_stack[i]

        frames_surroundings = []
        frame_target = []
        for (t_idx, t) in enumerate(range(t_start, t_end)):
            if t_idx == int((t_end-t_start) / 2):
                frame_target.append(self.acqs[acq_idx][t][:, x_start:x_end, y_start:y_end])
            else:
                frames_surroundings.append(self.acqs[acq_idx][t][:, x_start:x_end, y_start:y_end])

        return (frames_surroundings, frame_target, torch.tensor(patch_index_stich_patch), torch.tensor(patch_indices_stich_stack))