# supervised-behav-classification
Contains classification models designed for analyzing 3D input data, i.e. involving both spatial (2D) and temporal (time) components. The datasets of interest were the singly-housed mouse MIT and SCORHE videos. These models have been specifically trained and fine-tuned to classify and analyse complex behaviours in mouse video data.

## Models
All model files (hdf5/.h5) can be downloaded under the **assets** folder in release v0.0.1.0 

### MIT Models

- **Description:** The MIT models are convolutional neural networks (CNN) trained for mouse video classification.

### SCORHE-finetuned Models

- **Description:** The SCORHE models are MIT models fine-tuned on the SCORHE video dataset.

## Usage

To use these models in your projects, you can follow the steps below:

1. Download the desired model from the indicated directory.
2. Load the model into your TensorFlow/Keras machine learning framework.
3. Use the model's inference methods to perform spatiotemporal classification.

Although built for spatial dimensions of 128 by 128 only, the raw codes are also available as .py files to allow for fresh training and even model redesign.

