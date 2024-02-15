# Python Package for Direct Stochastic Optical Reconstruction Microscopy (dSTORM)

This Python package provides a comprehensive toolkit for performing direct Stochastic Optical Reconstruction Microscopy (dSTORM) analysis using Attention AI algorithms. dSTORM is a super-resolution microscopy technique that achieves high-resolution imaging beyond the diffraction limit by localizing fluorescent molecules.
Features

    Attention AI Localization: Utilize advanced Attention AI algorithms for accurate localization of fluorescent molecules.
    EMCCD Camera Simulator: Integrated simulator for accurate simulations of Electron Multiplying Charge-Coupled Device (EMCCD) cameras to mimic experimental conditions.
    OpenGL-based Visualization Tool: Interactive visualization tool based on OpenGL for visualizing dSTORM data and results.
    Training Script: Script for training Attention AI models using dSTORM datasets.
    Evaluation Script: Script for evaluating trained models on dSTORM datasets.

Installation

You can install the package via pip:

bash

pip install dstorm-attention-ai

Usage
Localization with Attention AI

python

from dstorm_attention_ai import localize

## Load dSTORM data
data = load_dstorm_data('path/to/data')

## Perform localization using Attention AI
localizations = localize(data)

EMCCD Camera Simulator

python

from dstorm_attention_ai.simulator import EMCCDCameraSimulator

## Initialize EMCCD camera simulator
simulator = EMCCDCameraSimulator()

## Simulate EMCCD camera imaging
simulated_image = simulator.simulate(data)

Visualization

python

from dstorm_attention_ai.visualization import visualize_dstorm_data

## Visualize dSTORM data
visualize_dstorm_data(data)

Training Script

bash

python train.py --dataset path/to/training_data --epochs 50 --batch_size 32

Evaluation Script

``
python evaluate.py --model path/to/model --dataset path/to/evaluation_data
``
Contributors

    Sebastian Reinhard (sebastian.uj.reinhard@gmail.com)
    Jann Schrama (@)
    Markus Sauer

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    This package was developed as part of research conducted at Julius Maximillians University.
    Hat tip to the open-source community for their support and contributions.