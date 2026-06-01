import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from cnn_model import build_cnn


IMG_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 10


# Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# Training Data
train_data = datagen.flow_from_directory(
    "dataset/prediction",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


# Validation Data
val_data = datagen.flow_from_directory(
    "dataset/prediction",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


# Build Model
model = build_cnn()


# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
import os

os.makedirs("saved_models", exist_ok=True)

# Save model
model.save("saved_models/cnn_prediction_model.h5")


print("✅ CNN Prediction Model Saved!")