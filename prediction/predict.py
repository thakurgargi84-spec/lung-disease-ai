import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# Image size
IMG_SIZE = 256


# Load trained CNN model
model = load_model("saved_models/cnn_prediction_model.h5")


# Class names
classes = [
    "COVID",
    "NORMAL",
    "PNEUMONIA",
    "TB"
]


# Image path
image_path = "test_image.png"

# Read image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Keep original
original = image.copy()


# Resize
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))


# Normalize
image = image / 255.0


# Expand dimensions
image = np.expand_dims(image, axis=-1)

image = np.expand_dims(image, axis=0)


# Predict
prediction = model.predict(image)


# Get class
predicted_class = np.argmax(prediction)


# Confidence
confidence = np.max(prediction) * 100


# Output
print("Predicted Disease:", classes[predicted_class])

print("Confidence:", confidence)


# Show image
plt.imshow(original, cmap="gray")

plt.title(
    f"{classes[predicted_class]} ({confidence:.2f}%)"
)

plt.axis("off")

plt.show()