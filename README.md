# MobileNetV2 for Fish vs No-Fish Classification

This repository contains the implementation of a fish vs no-fish classification model using MobileNetV2 in PyTorch. The code includes data preprocessing, model training, evaluation, and saving the model in multiple formats for deployment.

## Features
- **Data Augmentation:** Random flipping, rotation, color jitter, and Gaussian blur.
- **Model Architecture:** MobileNetV2 with customizable layers and dropout.
- **Performance Metrics:** Confusion matrix and final loss values saved to JSON and images.
- **Multiple Model Formats:** State dictionary, complete model, and TorchScript.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mobilenet_fish_detection.git
   cd mobilenet_fish_detection
