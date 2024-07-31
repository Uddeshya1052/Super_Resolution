<h1 align="center">Welcome to SRGAN On Custom Dataset 👋</h1>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

> **SRGAN On Custom Dataset**: Learn how to train SRGAN on a custom dataset to achieve high-quality image super-resolution.

---

### 🏠 [Homepage](https://github.com/yourusername/SRGAN-Custom-Dataset)

## Introduction

Welcome to the SRGAN project! This repository contains code and instructions for training a Super-Resolution Generative Adversarial Network (SRGAN) on a custom dataset. SRGAN is a powerful neural network architecture that can upscale low-resolution images to high-resolution images with impressive detail and fidelity.

## ✨ Features

- **Custom Dataset Support:** Train SRGAN on your custom dataset to achieve the best results for your specific use case.
- **Flexible Environment:** Easily set up and configure the environment using Anaconda and PyTorch.
- **GPU & CPU Support:** Train and test your model on both GPU and CPU hardware, depending on your resources.
- **Simple Training & Testing:** Straightforward commands for training and testing the SRGAN model.

## Prerequisites

Ensure you have the following prerequisites before proceeding with the installation:

- Anaconda
- Python 3.7+
- Conda

## 📦 Environment Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/SRGAN-Custom-Dataset.git
   cd SRGAN-Custom-Dataset

2. **Create and activate the conda environment:**

   Open the Anaconda prompt and navigate to the folder where you have your environment.yml file:
   ```sh
   conda env create -f environment.yml
   conda activate srganenv_gpu

3. **Set up the environment:**
   Depending on your hardware, choose the appropriate PyTorch installation:
   - **GPU**
      ```sh
        conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
  
   -  **CPU**
        ```sh
        conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch

## 🚀 Usage

Train Your Model:

To train your model on a custom dataset, execute the following command:

    python main.py --LR_path custom_dataset_cars/hr_train_LR --GT_path custom_dataset_cars/hr_train_HR

- **--LR_path:** Path to your low-resolution training images.
- **--GT_path:** Path to your ground truth high-resolution images.

Test Your Model

    python main.py --mode test_only --LR_path test_data/cars --generator_path ./model/srgan_custom.pt

  - `--mode test_only`: Set this flag to indicate testing mode.
  - `--LR_path`: Path to your low-resolution test images.
  - `--generator_path`: Path to your trained SRGAN generator model.

## 🤝 Contributing
   Contributions, issues, and feature requests are welcome! Feel free to check the issues page. You can 
   also take a look at the contributing guide.

## 👤 Author
  - Name: Uddeshya Srivastava
  - GitHub: [Uddeshya1052](https://github.com/Uddeshya1052/Super_Resolution)
  - LinkedIn: [Uddeshya Srivastava](https://www.linkedin.com/in/uddeshya-srivastava-739881137/)
  

