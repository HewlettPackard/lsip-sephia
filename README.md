# SEPhIA: <1 laser/neuron Spiking Electro-Photonic Integrated Multi-Tiled Architecture for Scalable Optical Neuromorphic Computing

This repository contains the code and trained models for the SEPhIA paper.

## Repository Structure

- **`picsim/`** - Minimal Python package for modelling of photonic integrated circuits (PICs)
  - Provides core components for the MRM-based weight banks, photodetectors, and opto-electronic layers
  - Contains platform definitions and utility scripts for optical neural network simulations

- **`sephia/`** - SEPhIA model implementation and training code
  - `SEPhIA_OE2.py` - Main SEPhIA multi-tiled opto-electronic SNN architecture
  - `OESNN_Runs/train_OESNN5.py` - Training script for O/E-SNN models
  - `OESNN_Runs/funcs_training.py` - Training utility functions
  - `utils.py` - Helper functions for checkpointing and data processing

- **`checkpoints/`** - Trained model checkpoints for different configurations
  - Organized by WDM frequency spacing (50.0GHz, 63.0GHz, 100.0GHz) and MRM parameter sets. 
  - Each includes 5 repeated runs, with single reference PyTorch checkpoint per run. 
  - Metadata and training run metrics are stored in the corresponding .json file. Sweeps over various optical power levels are included as .npy files (numpy arrays). 
  - The optical power levels are used using expected value at the final stage (therefore, for example, -12dBm in the script corresponds to 6dBm at optical source for the N_T=16 tile).

## Usade

Install the picsim package in your Python 3 environment (`-e` installs it in-place):

```bash
pip install -e picsim
```

Run the training procedure for the O/E-SNN:

```bash
cd sephia/OESNN_Runs
python train_OESNN5.py
```