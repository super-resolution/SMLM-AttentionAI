# SMLM-AttentionAI - Improving single molecule localisation microscopy reconstruction by extending the temporal context
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub issues](https://img.shields.io/github/issues/super-resolution/SMLM-AttentionAI.svg)](https://github.com/super-resolution/SMLM-AttentionAI/issues)


This repository hosts the code and data for SMLM-AttentionAI as described in

__"Improving single molecule localisation microscopy reconstruction by extending the temporal context"__
by Sebastian Reinhard, Vincent Ebert, Jann Schrama, Markus Sauer and Philip Kollmannsberger (2024)

----

SMLM-AttentionAI implements a novel deep learning method for single-molecule localization microscopy (SMLM) that harnesses extended temporal context to enhance localization precision. Traditional SMLM techniques typically fit Gaussian models to isolated frames, limiting their performance under high-density conditions and challenging signal-to-noise ratios. Our approach overcomes these limitations by:

- **Leveraging Temporal Information** by integrating long time-series data to capture repeated emitter activity and background fluctuations, improving localization in complex imaging scenarios.
- **Incorporating Attention Mechanisms** by combining a U-Net with a multi-head attention block inspired by Transformer models, enabling the network to dynamically focus on relevant temporal features.
- **Accurate Simulation** using an EMCCD-based simulator to generate realistic training data, ensuring that the model is well-calibrated to the specific noise and signal properties encountered in practical experiments.
- **Competitive Performance** demonstrating performance close to the theoretical Cramér-Rao lower bound and comparing favorably against state-of-the-art tools through extensive benchmarking and ablation studies.
- **Resource Efficiency** with a shallow spatial compression and a broad temporal context, making retraining feasible even in resource-constrained environments.

This project aims to advance SMLM imaging by fully exploiting the temporal correlations inherent in blinking fluorophores, thereby enabling more robust and precise super-resolution microscopy under a variety of experimental conditions.


## Installation


1. From source
```bash
git clone https://github.com/super-resolution/SMLM-AttentionAI.git
cd SMLM-AttentionAI
pip install -r requirements.txt
```

Ensure you have Python 3.9+ and pip installed on your system. We recommend using a dedicated environment, e.g. using miniconda.

## Usage
### Running an evaluation
To evaluate a dataset:
1. Configure evaluation parameters in `eval.yaml` and `default.yaml`:

    Key configuration options:
    - Network
    - Training (corresponding to network)
    - Dataset
    - Device

2. Run the evaluation script
    ```bash
    python run.py #for data without groundtruth
    ```
    The script generates:
    - Localisations from the underlying image data
    - An high performance OpenGL rendering instance
    - Hardware optimized filters with live rendering
    - An output image
    ```
    python eval.py #for data with groundtruth
    ```
    The script generates:
    - CRLB plots
    - Jaccard-Index and RMSE

### Running simulations

To generate synthetic SMLM data for training or testing:

1. Configure simulation parameters in `simulation_images.yaml`:

    Key configuration options:
    - Frame count and dimensions
    - PSF parameters
    - Background noise levels
    - Data augmentation settings
    - Output path and format

2. Run the simulation script:
    ```bash
    python simulate_images.py
    ```
    The script generates:
    - Raw frames with realistic noise
    - Ground truth localizations
    - Metadata for training

Monitor progress in the console output and check the logs directory for detailed simulation reports.

### Training the network
To train the network
1. Configure training parameters in `train.yaml`
Key configuration options:
    - Network
    - Training (corresponding to network)
    - Dataset (Training and Validation)
    - Device
    - Batch size

2. Run training script `train.py`
The script generates:
    - Output Metrics for `compare_network_performance.py`
    - training checkpoints 

## Configuration Files

The `cfg` folder contains YAML configuration files that control various aspects of the application:

#### `network/{name}.yaml` 
Defines the Attention AI model architecture:
- Number of attention heads
- Layer configurations
- Activation functions
- Model dimensions

#### `train.yaml`
Training-specific settings:
- Learning rate
- Batch size
- Number of epochs
- Loss functions
- Optimizer parameters
- Training dataset
- Validation dataset

#### `eval.yaml`
Dataset configuration:
- Input/output paths
- Used network
- Used training


#### `defaults.yaml`
Defines falbacks for `network`, `dataset` and `optimizer`:
To use a specific configuration:
```python
from hydra import compose, initialize

initialize(config_path="cfg")
cfg = compose(config_name="config")
```


## SMLM-AttentionAI Simulation Module

This module contains code for simulating Single Molecule Localization Microscopy (SMLM) data with various data augmentation capabilities.

### Key Components:

#### Data Generation
- `sauer_lab.py`: Generates point cloud data for lab logo simulation
- `random_locs.py`: Creates random localization data
- `background_structure.py`: Handles background image generation
- `markov_chain.py`: Implements Markov chain modeling

#### Data Augmentation
- `data_augmentation.py`: Contains DropoutBox class for data augmentation via random region dropout
- `noise_simulations.py`: Simulates various noise patterns

#### Core Simulation
- `simulator.py`: Main simulation class orchestrating the full data generation pipeline
- `structs.py`: Contains core data structures

#### Testing
- Functional tests for complex simulations and dataset visualization
- Unit tests for individual components like dropout boxes, PSF calculations, and random location generation

The module is designed for generating synthetic SMLM data with realistic noise characteristics and data augmentation capabilities, primarily using PyTorch for GPU acceleration.

## SMLM-AttentionAI Models module

The project consists of several key components organized into directories:

### Core Modules
- `activations.py`: Custom activation functions including GMM activations
- `layers.py`: Neural network layer implementations
- `loader.py`: Model loading utilities with transfer learning support
- `loss.py`: Loss function implementations
- `network.py`: Base network architecture
- `unet.py`: U-Net implementation
- `util.py`: Utility functions for image processing


### Vision Transformer (VIT) Models
- `VIT/`: Contains multiple iterations of Vision Transformer implementations
    - Attention U-Net variants (`attentionunet.py`, `attentionunetv2.py`)
    - Diffusion models (v1-v4)
    - Base VIT implementation (`base.py`)
    - Multiple VIT versions (v3-v10) with various improvements
    - Test implementations (`vitvtest.py`)

Each module is designed to work together for SMLM (Single-Molecule Localization Microscopy) analysis using attention-based deep learning approaches.

## SMLM-AttentionAI Visualization Module

This module handles OpenGL-based visualization and GUI components for SMLM data display.

### Core Components

#### OpenGL Rendering
- `visualization_open_gl.py`: Main OpenGL visualization implementation
- `visualization.py`: Base visualization interface
- `shader.py`: Shader program management and compilation
- `textures.py`: Texture handling utilities
- `buffers.py`: OpenGL buffer operations
- `objects.py`: 3D object management

#### GUI Implementation
- `widget.py`: Custom Qt widgets
- `gui/user_interface.py`: PyQt5-based user interface
- `gui/user_interface.ui`: Qt Designer UI definition

### Key Features

#### Shader Support
- GLSL shader compilation and linking
- Uniform variable management
- Matrix transformations
- Vertex/Fragment shader support

#### Interactive Controls
- Precision filtering
- Frame selection
- Probability thresholding
- Real-time visualization updates

The module provides high-performance rendering capabilities with an intuitive user interface for exploring SMLM datasets.
## Contributors

    Sebastian Reinhard (sebastian.uj.reinhard@gmail.com)
    Philip Kollmannsberger
    Jann Schrama
    Markus Sauer

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

This package was developed as part of research conducted at Julius Maximillians University.
Hat tip to the open-source community for their support and contributions.
