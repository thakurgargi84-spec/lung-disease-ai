import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from cnn_model import build_cnn


# Parameters
IMG_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 3


# Dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    "dataset/prediction",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


val_data = datagen.flow_from_directory(
    "dataset/prediction",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


# Candidate learning rates
learning_rates = [
    0.1,
    0.01,
    0.001,
    0.0001
]


best_accuracy = 0

best_lr = 0


# GA Loop
for lr in learning_rates:

    print(f"\nTesting Learning Rate: {lr}")

    model = build_cnn()

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1
    )

    accuracy = max(history.history["val_accuracy"])

    print(f"Validation Accuracy: {accuracy}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_lr = lr


print("\n========== FINAL RESULT ==========")

print(f"Best Learning Rate: {best_lr}")

print(f"Best Validation Accuracy: {best_accuracy}")