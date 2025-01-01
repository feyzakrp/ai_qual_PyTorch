# Image Classification with PyTorch

This repository contains a pipeline for training an image classification model using PyTorch. The project includes dataset preparation, data augmentation, model training, and evaluation, using the EfficientNet architecture for high performance.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview
This project demonstrates how to train an image classification model using PyTorch. Key features include:
- Dataset preprocessing with custom transformations.
- Transfer learning using EfficientNet.
- Validation and test set evaluation.
- GPU support for faster training.

---

## Dataset

The dataset is structured as follows:
- **Training Set**: Images for training the model.
- **Validation Set**: Images for model validation during training.
- **Test Set**: Images for evaluating the model’s performance.

Ensure the dataset is split into appropriate directories and labeled correctly. The labels for the test dataset are provided in `test.csv`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-classification-pytorch.git
   cd image-classification-pytorch
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a GPU available for training (optional but recommended).

---

## Usage

### Dataset Preparation
Ensure your dataset is structured and labeled properly. Update the paths in the code to point to your dataset directories:
- `train_dir`
- `val_dir`
- `test_dir`

---

## Model Architecture
This project uses EfficientNet, a state-of-the-art convolutional neural network for image classification. Key features:
- Pretrained weights for transfer learning.
- Modified final layer to match the number of target classes.

---

## Results
After training for 5 epochs:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Test Accuracy**: TBD (ensure your test data is correctly loaded and evaluated).

### Training Loss Curve
Loss decreases steadily across epochs without significant overfitting, as shown in the training loss curves.

---

## Future Work
- **Improve Test Data Handling**: Address the issue with test data loading.
- **Add More Metrics**: Include precision, recall, and F1-score.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizers.
- **Model Optimization**: Implement mixed-precision training for faster execution.
- **Visualization**: Use Grad-CAM for visualizing model predictions.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.




