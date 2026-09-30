# Image Classification with Transfer Learning

Image classification using transfer learning with a pretrained ResNet18 deep learning model.

## Overview

This project demonstrates image classification using **transfer learning** with a pretrained **ResNet18** convolutional neural network.

A pretrained ResNet18 model was adapted to classify images from the **CIFAR-10** dataset. The original classification layer was replaced with a new layer containing 10 output classes, while the pretrained layers were frozen during training.

The project covers dataset preparation, image preprocessing, transfer learning, model training, evaluation, class-wise accuracy analysis, and confusion matrix visualization.

## Dataset

The project uses the **CIFAR-10** dataset, which contains 60,000 color images across 10 classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

For this experiment, a subset of:

* **5,000 training images**
* **1,000 test images**

was used to keep training practical on CPU resources.

## Model

**ResNet18** pretrained on ImageNet was used as the base model.

The transfer learning process was:

1. Load the pretrained ResNet18 model.
2. Freeze the pretrained layers.
3. Replace the original final classification layer.
4. Configure the new layer for 10 CIFAR-10 classes.
5. Train the new classification layer on the CIFAR-10 training subset.
6. Evaluate the model on the test subset.

## Results

| Metric            |            Result |
| ----------------- | ----------------: |
| Training samples  |             5,000 |
| Test samples      |             1,000 |
| Model             |          ResNet18 |
| Training approach | Transfer Learning |
| Epochs            |                 1 |
| Training loss     |            0.9142 |
| Test accuracy     |            72.48% |

The model achieved **72.48% accuracy on the 1,000-image test subset** after one training epoch.

## Evaluation

Model performance was examined using:

* Overall test accuracy
* Class-wise accuracy
* Sample predictions
* Confusion matrix

These evaluations provide additional insight into how the model performs across the different CIFAR-10 classes.

## Technologies

* Python
* PyTorch
* Torchvision
* ResNet18
* CIFAR-10
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Limitations & Future Improvements

This experiment uses a subset of CIFAR-10 and trains only the final classification layer of ResNet18 for one epoch. The experiment was also performed using CPU resources.

Possible improvements include:

* Training on the complete CIFAR-10 training dataset
* Fine-tuning additional ResNet18 layers
* Training for multiple epochs
* Applying data augmentation
* Comparing multiple pretrained architectures
* Evaluating additional metrics such as precision, recall, and F1-score

## Notebook

The complete implementation is available in:

`Image_Classification_Transfer_Learning.ipynb`
