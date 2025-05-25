# Building Detection Using U-Net Architecture

A deep learning project for automated building detection from satellite/aerial imagery using a U-Net-like convolutional neural network architecture.

## Overview

This project implements a semantic segmentation model to detect buildings in satellite images. The model uses a U-Net-inspired architecture with encoder-decoder structure and skip connections to perform pixel-wise classification for building detection.

## Features

- **U-Net Architecture**: Custom implementation with encoder-decoder structure  
- **Skip Connections**: Preserves spatial information during upsampling  
- **Data Preprocessing**: Automated loading and preprocessing of training/testing data  
- **Model Checkpointing**: Saves the best model based on validation loss  
- **Visualization**: Built-in visualization tools for results analysis  
- **Binary Segmentation**: Outputs binary masks for building detection  

## Requirements
- tensorflow>=2.0
- numpy
- matplotlib
- Pillow
- glob
- os


## Installation

1. Clone this repository.
2. Install required dependencies:

    ```
    pip install tensorflow numpy matplotlib pillow
    ```

## Dataset Structure

Organize your data in the following structure:
```
BuildingDetection/
├── train/
│ ├── image1_image.tif
│ ├── image1_label.tif
│ ├── image2_image.tif
│ ├── image2_label.tif
│ └── ...
├── test/
│ ├── test1_image.tif
│ ├── test1_label.tif
│ ├── test2_image.tif
│ ├── test2_label.tif
│ └── ...
└── Building-Detection_Improvements.py
```


**Image Specifications:**
- Input images: 128x128x3 (RGB)
- Label masks: 128x128x1 (Binary)
- Format: TIFF files
- Naming convention: `*_image.tif` for images, `*_label.tif` for masks

## Usage

1. **Update the working directory** in the script:

    ```
    os.chdir(r'C:\Users\suyas\Desktop\BuildingDetection')
    ```

2. **Run the complete pipeline**:

    ```
    python Building-Detection_Improvements.py
    ```

The script will automatically:
- Load and preprocess the data
- Build the U-Net model
- Train for 100 epochs
- Save the best model as `best_model.h5`
- Generate predictions and visualizations

## Model Architecture

### Encoder (Downsampling Path)
- **Block 1**: Conv2D(32) → Dropout(0.25) → Conv2D(32) → MaxPool2D  
- **Block 2**: Conv2D(32) → Dropout(0.25) → Conv2D(32) → MaxPool2D  
- **Block 3**: Conv2D(64) → Dropout(0.25) → Conv2D(64) → MaxPool2D  
- **Bottleneck**: Conv2D(64) → Dropout(0.5) → Conv2D(64)

### Decoder (Upsampling Path)
- **Block 1**: Conv2DTranspose(64) → Dropout(0.5) → Upsample → Concatenate  
- **Block 2**: Conv2DTranspose(64) → Dropout(0.5) → Upsample → Concatenate  
- **Block 3**: Conv2DTranspose(32) → Dropout(0.5) → Upsample → Concatenate  
- **Output**: Conv2D(32) → Conv2D(32) → Conv2D(1, sigmoid)

### Training Configuration
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Epochs**: 100  
- **Batch Size**: 10  
- **Validation**: Test set  

## Output Files

The script generates several output files:

- `train_xx.npy` - Preprocessed training images  
- `train_yy.npy` - Preprocessed training masks  
- `test_xx.npy` - Preprocessed test images  
- `test_yy.npy` - Preprocessed test masks  
- `best_model.h5` - Best trained model weights  

## Results

The model outputs:
- **Training/Validation Loss Curves**: Visual plots showing model performance  
- **Binary Predictions**: Thresholded predictions (threshold = 0.5)  
- **Comparison Visualizations**: Side-by-side predicted vs actual masks  

## Model Performance

- **Monitoring**: Validation loss is monitored for early stopping  
- **Threshold**: 0.5 for binary classification  
- **Evaluation**: Visual comparison of predicted and actual building masks  

## Customization

### Adjusting Hyperparameters

Training parameters
- epochs = 100 # Number of training epochs
- batch_size = 10 # Batch size for training
- threshold = 0.5 # Prediction threshold

Model parameters
- dropout_rate = 0.25 # Dropout rate for regularization


### Model Architecture Modifications

- Adjust filter sizes in Conv2D layers  
- Modify dropout rates for different regularization  
- Change activation functions  
- Add batch normalization layers  

## Troubleshooting

**Common Issues:**
1. **Path Error**: Ensure the working directory path is correct  
2. **Memory Error**: Reduce batch size if running out of GPU memory  
3. **File Format**: Ensure images are in TIFF format with correct naming  
4. **Image Size**: All images must be 128x128 pixels  

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Submit a pull request  

## Contact

For questions or issues, please open an issue in the repository.





