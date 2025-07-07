# LIBRAS Sign Classifier with PyTorch and Transformers

## 📖 Description

This project implements a Deep Learning model for recognizing and classifying isolated signs from Brazilian Sign Language (LIBRAS) using video inputs. Using an *end-to-end* architecture, the model directly analyzes raw video pixels for classification, with a focus on high accuracy and computational efficiency.

The work was developed as part of a scientific paper, comparing different neural network architectures such as 3D-CNNs (I3D) and Video Transformers.

### ✨ Features

* **High Accuracy:** Achieves over 90% accuracy on the test set.
* **End-to-End Pipeline:** Directly processes raw RGB videos.
* **Robust Validation:** The test set is composed of signers unseen during training, ensuring the model's generalization capabilities.
* **Reproducibility:** The project is fully containerized with Docker, ensuring anyone can run the training with just two commands.

## 🛠️ Tech Stack

* Python 3.10+
* PyTorch
* FFmpeg
* Conda
* Docker
* NVIDIA CUDA

## ⚙️ Project Structure

.
├── dataset_splits/     # CSV files with train, validation, and test splits
├── training_plots/       # Folder where training plots are saved
├── build_pytorch_model.py # Main script for model training and evaluation
├── extract_metadata.py    # Script to extract metadata from videos
├── split_dataset.py       # Script to perform the dataset split
├── minds_libras_metadata.csv # Dataset metadata
├── Dockerfile             # Recipe to build the project's Docker image
├── environment.yml        # List of Conda environment dependencies
└── README.md              # This documentation


## 🚀 Installation and Usage

You can run this project in two ways: the recommended method (with Docker) or the manual method.

### Method 1: Using Docker (Recommended)

**Prerequisites:** [Docker](https://www.docker.com/get-started) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support).

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/VictorCezar/end_to_end_LIBRAS.git](https://github.com/VictorCezar/end_to_end_LIBRAS.git)
    cd end_to_end_LIBRAS
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t libras-classifier .
    ```

3.  **Run the training:**
    ```bash
    docker run --gpus all -v "$(pwd)/dataset_splits:/app/dataset_splits" -v "$(pwd)/training_plots:/app/training_plots" libras-classifier
    ```
    The plots and training results will appear in the `training_plots` folder.

### Method 2: Manually with Conda

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/VictorCezar/end_to_end_LIBRAS.git](https://github.com/VictorCezar/end_to_end_LIBRAS.git)
    cd end_to_end_LIBRAS
    ```

2.  **Create the Conda environment from the environment file:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate libras_env  # Replace 'libras_env' with your environment's name
    ```

4.  **Run the training script:**
    ```bash
    python build_pytorch_model.py
    ```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue to discuss improvements.