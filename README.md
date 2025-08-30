# QFLIM: Quantum-Aware First-Photon FLIM

QFLIM is a self-supervised denoising method for Fluorescence Lifetime Imaging Microscopy (FLIM) that fully exploits the quantum nature of fluorescence emission. It reduces photon demand by over three orders of magnitude, enabling high-fidelity FLIM even under extreme low-light conditions. QFLIM shows strong robustness to intensity fluctuations, making it ideal for deep-tissue imaging and capturing fast biological dynamics with minimal phototoxicity.

## Overview

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