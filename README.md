# supervised-behav-classification
Contains classification models designed for analyzing 3D input data, i.e. involving both spatial (2D) and temporal (time) components. The datasets of interest were the singly-housed mouse MIT and SCORHE videos. These models have been specifically trained and fine-tuned to classify and analyse complex behaviours in mouse video data.

## Models

### MIT Models

- **Description:** The MIT models are convolutional neural networks (CNN) trained for image transformation tasks.
- **Usage:** Transfer learning to other ST research animal data, behavioural phenotyping, and ethogramming.
- **Download:** You can download the MIT model from the MIT folder. 

### SCORHE-finetuned Models

- **Description:** The SCORHE-finetuned model is a variant of the SCORHE fine-tuned on the SCORHE video dataset for improved performance.
- **Usage:** Transfer learning to other ST research animal data, behavioural phenotyping, and ethogramming.
- **Download:** You can download the SCORHE-finetuned model from the SCORHE folder.

## Usage

To use these models in your projects, you can follow the steps below:

1. Download the desired model from the provided directories.
2. Load the model into your TensorFlow/Keras machine learning framework.
3. Use the model's inference methods to perform spatiotemporal classification.


