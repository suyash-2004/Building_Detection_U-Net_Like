import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout

# --- Step 1: Set the working directory ---
os.chdir(r'C:\Users\suyas\Desktop\BuildingDetection')

# --- Step 2: Load training and testing image/mask paths ---
train_x = sorted(glob.glob('train/*_image.tif'))
train_y = sorted(glob.glob('train/*_label.tif'))
test_x = sorted(glob.glob('test/*_image.tif'))
test_y = sorted(glob.glob('test/*_label.tif'))

print(f"Training samples: {len(train_x)}, Testing samples: {len(test_x)}")

# --- Step 3: Convert training images to NumPy arrays ---
train_xx = np.zeros((len(train_x), 128, 128, 3))
train_yy = np.zeros((len(train_y), 128, 128, 1))

for i, (img_path, mask_path) in enumerate(zip(train_x, train_y)):
    img = Image.open(img_path)
    train_xx[i] = np.array(img)

    mask = Image.open(mask_path)
    train_yy[i] = np.array(mask).reshape(128, 128, 1)

# --- Step 4: Convert testing images to NumPy arrays ---
test_xx = np.zeros((len(test_x), 128, 128, 3))
test_yy = np.zeros((len(test_y), 128, 128, 1))

for i, (img_path, mask_path) in enumerate(zip(test_x, test_y)):
    img = Image.open(img_path)
    test_xx[i] = np.array(img)

    mask = Image.open(mask_path)
    test_yy[i] = np.array(mask).reshape(128, 128, 1)

# --- Step 5: Display sample image and mask ---
plt.imshow(train_xx[0].astype('uint8'))
plt.title("Sample RGB Image")
plt.show()

plt.imshow(train_yy[0, :, :, 0].astype('uint8'), cmap='gray')
plt.title("Corresponding Label (Mask)")
plt.show()

# --- Step 6: Save arrays for future use ---
np.save('train_xx.npy', train_xx)
np.save('train_yy.npy', train_yy)
np.save("test_xx.npy", test_xx)
np.save("test_yy.npy", test_yy)

# --- Step 7: Build the "U-Net-like" model ---

x_in = Input(shape=(128, 128, 3))

# Encoder
x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
x_temp = Dropout(0.25)(x_temp)
x_skip1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2, 2))(x_skip1)

x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2, 2))(x_skip2)

x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2, 2))(x_skip3)

x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)

# Decoder
x_temp = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip3])

x_temp = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip2])

x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip1])

x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_temp)

# Output layer
x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_out = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x_temp)

# Compile model with binary crossentropy loss
model = Model(inputs=x_in, outputs=x_out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- Save best model callback ---
checkpoint = ModelCheckpoint(
    'best_model.h5',        # file to save best model
    monitor='val_loss',     # monitor validation loss
    verbose=1,
    save_best_only=True,    # save only when val_loss improves
    mode='min'
)

# --- Train the model ---
history = model.fit(
    train_xx, train_yy,
    validation_data=(test_xx, test_yy),
    epochs=100,
    batch_size=10,
    verbose=1,
    callbacks=[checkpoint]
)

# --- Plot training and validation loss ---
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# --- Predict on test set ---
threshold = 0.5
pred_test = model.predict(test_xx)
pred_test = (pred_test > threshold).astype(np.uint8)
print(f"Prediction shape: {pred_test.shape}")

# --- Visualize a random prediction vs actual mask ---
index = 10
plt.subplot(1,2,1)
plt.title('Predicted Mask')
plt.imshow(pred_test[index, :, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Actual Mask')
plt.imshow(test_yy[index, :, :, 0], cmap='gray')
plt.axis('off')

plt.show()
