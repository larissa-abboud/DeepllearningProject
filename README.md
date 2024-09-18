# Beans Disease Classification Using CNN

## Overview

This project presents the development, implementation, and evaluation of a Convolutional Neural Network (CNN) model for classifying images from the "Beans" dataset. The dataset contains field-captured images of beans categorized into three classes: Angular Leaf Spot, Bean Rust, and healthy beans. The objective is to leverage deep learning techniques to accurately identify these conditions, contributing to better disease management in agriculture.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction

Image classification is a critical tool in various fields, particularly in agriculture, where it facilitates early disease detection and management. The "Beans" dataset offers a comprehensive collection of images showcasing beans affected by Angular Leaf Spot and Bean Rust, as well as healthy specimens. This study focuses on developing a CNN model that can effectively classify these diseases using deep learning techniques.

## Dataset

The "Beans" dataset consists of images captured using smartphone cameras and has been meticulously annotated by experts from the National Crops Resources Research Institute (NaCRRI) in Uganda. The dataset is categorized into three distinct classes:

- Angular Leaf Spot
- Bean Rust
- Healthy Beans

This structure makes it suitable for training and evaluating machine learning models focused on disease detection.

## Methodology

The project involved several key steps:

1. **Data Preprocessing**: Images were resized, normalized, and augmented to enhance model training.
2. **Model Construction**: Two CNN models were developed:
   - A standard CNN architecture.
   - A modified VGG16 architecture.
3. **Training**: Each model was trained on the dataset with specified epochs.
4. **Evaluation**: Model performance was assessed using test accuracy metrics.

## Model Architectures

1. **Standard CNN**: This model followed a conventional CNN architecture with multiple convolutional and pooling layers.
   - **Achieved Test Accuracy**: 82.03% after training for 100 epochs.

2. **VGG16-based CNN**: This model utilized the VGG16 architecture, which is known for its deep layers and effectiveness in image classification.
   - **Achieved Test Accuracy**:
     - 84% after training for 10 epochs.
     - Approximately 82% after 17/50 and 5/50 epochs.

## Results

The performance of the models demonstrated the effectiveness of deep learning in classifying bean diseases. The VGG16-based model outperformed the standard CNN, achieving the highest test accuracy of 84%.

## Conclusion

This project successfully developed and evaluated two CNN models for the classification of bean diseases using the "Beans" dataset. The results indicate that deep learning techniques can significantly aid in the early detection of agricultural diseases, thereby enhancing disease management strategies.

## Installation

To run this project, ensure you have the following prerequisites:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV (optional for image processing)


