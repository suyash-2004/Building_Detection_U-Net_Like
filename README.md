# Building Detection with U-Net-like Model

A deep learning pipeline for building detection in satellite imagery using a custom U-Net-like architecture with TensorFlow/Keras.

---

## **Features**

- Loads and preprocesses satellite images and masks for training and testing
- Implements a U-Net-inspired convolutional neural network for semantic segmentation
- Includes data visualization for sample images and masks
- Trains the model with validation monitoring and checkpointing
- Visualizes training/validation loss and prediction results

---

## **Project Structure**

.
├── Building-Detection_Improvements.py
├── train/
│ ├── *_image.tif
│ └── *_label.tif
├── test/
│ ├── *_image.tif
│ └── *_label.tif
├── train_xx.npy
├── train_yy.npy
├── test_xx.npy
├── test_yy.npy
└── best_model.h5


---

## **Setup**

1. **Clone the repository**

.
├── Building-Detection_Improvements.py
├── train/
│ ├── *_image.tif
│ └── *_label.tif
├── test/
│ ├── *_image.tif
│ └── *_label.tif
├── train_xx.npy
├── train_yy.npy
├── test_xx.npy
├── test_yy.npy
└── best_model.h5

text

---

## **Setup**

1. **Clone the repository**

git clone <repo_url>
cd <repo_folder>


2. **Install dependencies**

pip install numpy matplotlib pillow tensorflow


3. **Prepare your data**

- Place training images and masks in the `train/` directory.
- Place testing images and masks in the `test/` directory.
- Images should be named as `*_image.tif` and corresponding masks as `*_label.tif`.

---

## **Usage**

Run the main script:



