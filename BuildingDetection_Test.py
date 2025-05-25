import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('best_model.h5')

# Load test data
test_xx = np.load('test_xx.npy')
test_yy = np.load('test_yy.npy')

# Predict on test data
threshold = 0.5
predictions = model.predict(test_xx)
predictions = (predictions > threshold).astype(np.uint8)

# Visualize predictions vs actual masks
num_samples = len(test_xx)
for i in range(num_samples):
    plt.figure(figsize=(8, 4))

    # Predicted mask
    plt.subplot(1, 2, 1)
    plt.title(f'Predicted Mask {i}')
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

    # Actual mask
    plt.subplot(1, 2, 2)
    plt.title(f'Actual Mask {i}')
    plt.imshow(test_yy[i, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
