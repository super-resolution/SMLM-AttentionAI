# SMLM Attention AI

This Python package provides a comprehensive toolkit for performing direct Stochastic Optical Reconstruction Microscopy (dSTORM) analysis using Attention AI algorithms. dSTORM is a super-resolution microscopy technique that achieves high-resolution imaging beyond the diffraction limit by localizing fluorescent molecules.
Features
Attention AI Localization: Utilize advanced Attention AI algorithms for accurate localization of fluorescent molecules.
EMCCD Camera Simulator: Integrated simulator for accurate simulations of Electron Multiplying Charge-Coupled Device (EMCCD) cameras to mimic experimental conditions.
OpenGL-based Visualization Tool: Interactive visualization tool based on OpenGL for visualizing dSTORM data and results.
Training Script: Script for training Attention AI models using simulated datasets.
Evaluation Script: Script for evaluating trained models on dSTORM datasets.

## Installation

You can install the package via github:

````
github clone attention-ai
````
`cd` into the source code folder and install the necessary packages with:
````
pip install requirements.txt
````
## Usage
Describe yaml configuration fiels

### Load dSTORM data
Describe yaml

### Changing the defined model
localizations = localize(data)

EMCCD Camera Simulator

python

from dstorm_attention_ai.simulator import EMCCDCameraSimulator

### Simulate EMCCD camera imaging

````
main.py
````

### Visualize dSTORM data
visualize_dstorm_data(data)

Training Script


````
train.py
````

## Evaluation Script

````
run.py
````
## Contributors

    Sebastian Reinhard (sebastian.uj.reinhard@gmail.com)
    Jann Schrama (@)
    Markus Sauer

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

This package was developed as part of research conducted at Julius Maximillians University.
Hat tip to the open-source community for their support and contributions.