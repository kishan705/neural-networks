# Simple ResNet on CIFAR-10 (PyTorch)

This project implements a ResNet-like Convolutional Neural Network from scratch using PyTorch, trained on the CIFAR-10 dataset. It includes residual blocks, mixed precision training, and supports CPU, CUDA (GPU), and Apple Silicon (MPS).

---

## Features

- Custom implementation of Residual Blocks  
- Lightweight ResNet-like model for CIFAR-10  
- Mixed precision training for faster performance (on CUDA)  
- Supports CUDA, Apple M1/M2/M3/M4 (MPS), and CPU  
- CIFAR-10 image classification  
- Automatically saves model after training  

---

## Model Architecture

- 3 Convolutional Layers with Residual Connections  
- Adaptive Average Pooling  
- Fully Connected Output Layer  
- Batch Normalization and ReLU Activation  

---

## Tech Stack

- Python  
- PyTorch  
- torchvision  
- tqdm  

---

## Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/your-username/resnet-cifar10-pytorch.git
   cd resnet-cifar10-pytorch
