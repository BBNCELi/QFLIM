# QFLIM: Quantum-Aware First-Photon FLIM
> **QFLIM**: A self-supervised, quantum-aware deep learning method for ultra-low-light fluorescence lifetime imaging.

## Introduction

QFLIM (Quantum-Aware First-Photon FLIM) is a self-supervised deep learning method for fluorescence lifetime imaging microscopy (FLIM). Unlike conventional approaches that rely on photon histograms, QFLIM treats each excitation event as a quantum binary process — either no photon is emitted, or the first-arrival photon is detected with precise timing. By leveraging spatial and temporal context, QFLIM achieves accurate lifetime estimation under extremely low-light conditions, reducing photon demand by over three orders of magnitude. This enables fast, high-fidelity intravital imaging with strong robustness to intensity fluctuations, making QFLIM a powerful tool for studying dynamic biological processes in deep tissue.


QFLIM is a self-supervised denoising method for Fluorescence Lifetime Imaging Microscopy (FLIM) that fully exploits the quantum nature of fluorescence emission. It reduces photon demand by over three orders of magnitude, enabling high-fidelity FLIM even under extreme low-light conditions.

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