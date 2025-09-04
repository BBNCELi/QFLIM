# QFLIM
> A self-supervised, quantum-aware deep learning method for FLIM under extreme low light.

## Introduction
QFLIM (Quantum-Aware First-Photon FLIM) is a self-supervised deep learning method for fluorescence lifetime imaging microscopy (FLIM). Unlike conventional approaches that rely on photon histograms, QFLIM treats each excitation event as a quantum binary process — either no photon is emitted, or a single first-arrival photon is detected with precise timing. By leveraging spatial and temporal context, QFLIM achieves accurate lifetime estimation under extremely low-light conditions, reducing photon demand by over three orders of magnitude. This enables fast, high-fidelity intravital imaging with strong robustness to intensity fluctuations. QFLIM opens new opportunities for studying dynamic biological processes in deep tissue.

## Overview
- **Quantum-aware representation**: avoids histogram construction and directly models each excitation event.  
- **Self-supervised learning**: robust training without the need for paired datasets or ground-truth lifetimes.  
- **Extreme low light capability**: accurate lifetime estimation even below 1 photon per pixel (PPP).
- **Robustness to artifacts**: stable performance despite photobleaching, motion, or intensity fluctuations.  
- **Broad applicability**: enables fast, minimally invasive imaging of dynamic molecular processes in vivo.

## Workflow

### 1. Photon-arrival dataset preparation
- Rearrange raw photon data into **a sequence of frames that contain the arrival times of all photons**.
- For **Becker & Hickl** systems: convert `.SPC` files into `.tif` format.  
- For **PicoQuant** systems: convert `.PTU` files into `.tif` format.  
- We recommend using **at least 500 frames** with **PPP > 0.1** in regions of interest.  

### 1.1. (Optional) Simulation (if raw data are unavailable)
If you do not have experimental raw data, you can generate synthetic photon arrivals using the provided MATLAB scripts.  

Example:  
```
./0_simulations/run_simu_USAF1951.m
```

This script simulates **500 frames** with **PPP = 0.5** and saves the dataset to:
```
./simu_USAF1951_PPP0.5
```

Inside this folder, you will find:
- **Photon-arrival frames** : ./simu_USAF1951_PPP0.5/raw/frame*.tif  
- **Ground truth**: ./simu_USAF1951_PPP0.5/lt_gt/lt_gt.tif
- **FastFLIM** (center-of-mass method, CMM):  ./simu_USAF1951_PPP0.5/lt_fastflim/lt_fastflim.tif
- **Intensity-weighted lifetime visualizations**, saved with the default [colormap](https://uigradients.com/) suffix `_weddingdayblues` (e.g., ./simu_USAF1951_PPP0.5/lt_fastflim/lt_fastflim_lt500-3500_in0-0.5_weddingdayblue.tif)


### 2. Python training and inference
Once the dataset is prepared, you can train and evaluate QFLIM using the provided Python code.

#### Requirements
- **Python ≥ 3.9**  
- **GPU support** (CUDA-enabled GPU recommended)  
- **Additional Python packages**:  
  - numpy  
  - scipy  
  - tifffile  
  - tqdm  
  - matplotlib  

### Create a new conda environment (recommended):
```bash
conda create -n qflim python=3.10 -y
conda activate qflim
```

### GPU support
To run QFLIM efficiently on a GPU, make sure you have a working CUDA toolkit installed.
The recommended way is to install PyTorch together with the matching CUDA version directly from the [official PyTorch website](https://pytorch.org/get-started/locally/).

For example, on a machine with **CUDA 11.8**, you can install PyTorch with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Training example:
```bash
python run_QFLIM.py \
  --folderName .//simu_USAF1951_PPP0.5//raw
```

If you want to train on a specific GPU (e.g., GPU 2):
```bash
CUDA_VISIBLE_DEVICES=2 \
python run_QFLIM.py \
  --folderName .//simu_USAF1951_PPP0.5//raw
```

This script will:
- Read the raw data
- Perform training and inference for both lifetime and intensity
- Generate **lifetime video**, **intensity video** and **intensity-weighted lifetime visualization** as outputs