import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# =====================================
# SETTINGS
# =====================================

IMG_SIZE = 256

IMAGE_FOLDER = "dataset/segmentation/images"

FILENAME = "CHNCXR_0100_0.png"


# =====================================
# LOAD MODEL
# =====================================

print("Loading model...")

model = load_model(
    "saved_models/segmentation_model.h5",
    compile=False
)

print("Model loaded successfully!")


# =====================================
# LOAD IMAGE
# =====================================

image_path = os.path.join(
    IMAGE_FOLDER,
    FILENAME
)

print("Image:", image_path)

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:

    print("ERROR: Image not found!")
    print(image_path)

    exit()

original = image.copy()


# =====================================
# PREPROCESS
# =====================================

image = cv2.resize(
    image,
    (IMG_SIZE, IMG_SIZE)
)

image = image.astype(
    np.float32
) / 255.0

image = np.expand_dims(
    image,
    axis=-1
)

image = np.expand_dims(
    image,
    axis=0
)


# =====================================
# PREDICT MASK
# =====================================

print("Predicting...")

prediction = model.predict(
    image,
    verbose=0
)[0]

prediction = (
    prediction > 0.5
).astype(np.uint8)

prediction = prediction.squeeze()


# =====================================
# DISPLAY RESULTS
# =====================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original X-ray")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(prediction, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")

overlay = cv2.resize(
    prediction * 255,
    (original.shape[1], original.shape[0])
)

plt.imshow(original, cmap="gray")
plt.imshow(
    overlay,
    cmap="jet",
    alpha=0.4
)
plt.axis("off")

plt.tight_layout()
plt.show()