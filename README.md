# 🧠 miniFlow: A Deep Learning Framework Built from Scratch with NumPy

[![Built with Python](https://img.shields.io/badge/Python-3.8+%20|%20NumPy-blue?style=flat-square&logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Supported Parameter Count](https://img.shields.io/badge/Parameters-~100K-orange?style=flat-square)](https://github.com/alitkbbl/miniFlow)

## About miniFlow ✨

**miniFlow** is a lightweight, object-oriented, and educational deep learning framework implemented solely using the **NumPy** library, inspired by the **TensorFlow/Keras** syntax. The primary goal of this project is to gain a deeper understanding of the fundamental concepts of neural networks, including gradient computation (Backpropagation), standard layers, and optimization methods, by building them *from scratch*.

miniFlow serves as an excellent tool for developers and students who wish to explore the inner workings of larger deep learning frameworks.

## Key Features 🔑

Focusing on code clarity and efficiency (while maintaining implementation simplicity), miniFlow offers a comprehensive set of tools:

*   **Object-Oriented and Modular Architecture:** Structured with separate modules for layers, activation functions, and loss functions for easier maintenance and extension.
*   **Support for Advanced Layers:** Implementation of essential layers for modern neural networks:
    *   **Flatten Layer**
    *   **BatchNormalization Layer**
    *   **Dropout Layer**
*   **Powerful Callbacks System:** Includes an **Early Stopping** system (with `val_loss` monitoring capability) to prevent overfitting during training.
*   **Basic and Advanced Optimizers:** Support for key optimization methods:
    *   **SGD** (Stochastic Gradient Descent)
    *   **Adam**
*   **Essential Loss Functions:** Includes **SoftmaxCrossEntropy** (suitable for multi-class classification) and **MSE** (Mean Squared Error).
*   **Efficient Training:** Complete implementation of the four main stages of each epoch (Forward Pass, Loss Calculation, Backward Pass, Weight Update) with support for **Mini-Batching** for acceptable performance.

## Installation and Setup 🛠️

Since miniFlow is an educational project and its only dependency is NumPy, installation is straightforward:

### Prerequisites

*   Python 3.8+
*   NumPy
*   Matplotlib

## Documentation & Development 📚

For a deeper guide on the implementation of the Forward/Backward Pass in each layer and the mathematical details behind the optimizers, please refer to the files within the respective folders.





