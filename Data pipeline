import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image # For loading images
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. Simulate Data Collection and Organization ---
# Create dummy image files and a directory structure
data_dir = 'image_dataset'
if os.path.exists(data_dir):
    shutil.rmtree(data_dir) # Clean up previous runs
os.makedirs(data_dir, exist_ok=True)

classes = ['cat', 'dog', 'bird']
num_images_per_class = 50

print(f"Creating dummy image dataset in '{data_dir}'...")
for cls in classes:
    class_dir = os.path.join(data_dir, cls)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(num_images_per_class):
        # Create a dummy image (e.g., a small black image)
        dummy_image = Image.new('RGB', (64, 64), color = 'black')
        dummy_image.save(os.path.join(class_dir, f'{cls}_{i:03d}.png'))
print("Dummy dataset created.")

# --- 2. Data Preprocessing and Transformation Pipeline ---

# Step 2.1: Load image paths and labels using Pandas
image_paths = []
image_labels = []
for cls_idx, cls_name in enumerate(classes):
    class_path = os.path.join(data_dir, cls_name)
    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        image_labels.append(cls_name)

df = pd.DataFrame({'path': image_paths, 'label': image_labels})
print("\nDataFrame of image paths and labels:")
print(df.head())
print(f"Total images: {len(df)}")

# Map string labels to numerical IDs
label_to_id = {name: i for i, name in enumerate(classes)}
df['label_id'] = df['label'].map(label_to_id)
print("\nDataFrame with numerical labels:")
print(df.head())

# Step 2.2: Split data into training, validation, and test sets using Scikit-learn
X = df['path'].values
y = df['label_id'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
) # 0.25 of 0.8 is 0.2, so 60% train, 20% val, 20% test

print(f"\nTrain set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Step 2.3: Define a function to load and preprocess images for TensorFlow
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def load_and_preprocess_image(image_path, label):
    """Loads an image, resizes it, and normalizes pixel values."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3) # Use decode_png for .png files
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0 # Normalize to [0, 1]
    return img, label

# Step 2.4: Data Augmentation (Transformation)
def augment_image(image, label):
    """Applies random transformations for data augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Add more augmentations as needed (e.g., random_zoom, random_rotation)
    return image, label

# --- 3. Data Loading Pipeline (tf.data.Dataset) ---

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Apply preprocessing and augmentation
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE) # Apply augmentation to training data
train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\nTensorFlow Datasets created for efficient loading.")

# Verify a batch from the training dataset
for images, labels in train_ds.take(1):
    print(f"Shape of a training image batch: {images.shape}")
    print(f"Shape of a training label batch: {labels.shape}")
    break

# --- 4. Placeholder Deep Learning Model (TensorFlow Keras) ---
# This is a simple CNN, as requested in the previous turn.

num_classes = len(classes)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Use softmax for multi-class classification
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # from_logits=False because of softmax
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model (using the prepared datasets) ---
print("\nStarting model training with the prepared data pipeline...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5 # Reduced epochs for demonstration
)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Clean up dummy data
shutil.rmtree(data_dir)
print(f"\nCleaned up dummy dataset directory: '{data_dir}'")
