# QFLIM
> A self-supervised, quantum-aware deep learning method for FLIM under extreme low light.

## Introduction
QFLIM (Quantum-Aware First-Photon FLIM) is a self-supervised deep learning method for fluorescence lifetime imaging microscopy (FLIM). Unlike conventional approaches that rely on photon histograms, QFLIM treats each excitation event as a quantum binary process — either no photon is emitted, or a single first-arrival photon is detected with precise timing. By leveraging spatial and temporal context, QFLIM achieves accurate lifetime estimation under extremely low-light conditions, reducing photon demand by over three orders of magnitude. This enables fast, high-fidelity intravital imaging with strong robustness to intensity fluctuations. QFLIM opens new opportunities for studying dynamic biological processes in deep tissue.

## Overview
- **Quantum-aware representation**: avoids histogram construction and directly models each excitation event.  
- **Self-supervised learning**: robust training without the need for paired datasets or ground-truth lifetimes.  
- **Extreme low light capability**: accurate lifetime estimation even below 1 photon per pixel (PPP).
- **Robustness to artifacts**: stable performance despite photobleaching, motion, or segnsitivity fluctuations.  
- **Broad applicability**: enables fast, minimally invasive imaging of dynamic molecular processes in vivo.

## Workflow

### 1. Photon-arrival dataset preparation
- Rearrange raw photon data into a sequence of photon-arrival-time frames.  
- For **Becker & Hickl** systems: convert `.SPC` files into `.tif` format.  
- For **PicoQuant** systems: convert `.PTU` files into `.tif` format.  
- We recommend using **at least 500 frames** with **PPP > 0.5** in regions of interest.  

### 1.1. (Optional) Simulation (if raw data are unavailable)
If you do not have experimental raw data, you can generate synthetic photon arrivals using the provided MATLAB scripts.  

Example:  
```matlab
./0_simulations/run_simu_USAF1951.m

This script simulates **500 frames** with **PPP = 1** and saves the dataset to:
./simu_USAF1951_PPP1

Inside this folder, you will find:
- **Photon-arrival dataset** : ./simu_USAF1951_PPP1/frame*.tif  
- **Ground truth**: ./simu_USAF1951_PPP1/lt_gt/lt_gt.tif
- **fastflim** (Center of Mass Method, CMM):  ./simu_USAF1951_PPP1/lt_fastflim/lt_fastflim.tif
- **Intensity-weighted lifetime visualization**, saved with the suffix `_RGB` (e.g., ./simu_USAF1951_PPP1/lt_fastflim/lt_fastflim_RGB.tif)  

1. Arrival time of single photons
Rearrange the captured data into a series of photon-arrival-time frames. For Becker % Hickle, you should convert SPC file to tiffs. For PicoQuant, you should convert . We recomend at least 500 frames with PPP > 0.5.

If you do not have the raw data, you can simulate photon arrivals using Matlab codes as we provided. Run "run_simu_USAF1951.m" to simulate 500 frames with PPP = 1.

- **QFLIM** interprets each photon excitation event as a quantum binary process (either no photon or a single first-arrival photon).
- It effectively leverages spatial-temporal information to denoise and accurately extract fluorescence lifetime even when photon budgets are extremely low.
- QFLIM has been demonstrated to capture transient intracellular dynamics of ligand-dependent molecular states and multiplexed imaging of lymphocyte interactions during germinal center response.

## Installation

### Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

### Install Dependencies
To install all the required dependencies, run the following:

```bash
pip install -r requirements.txt