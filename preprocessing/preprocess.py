import cv2
import numpy as np
import os

IMG_SIZE = 256


# Preprocess image
def preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    # Normalize
    image = image / 255.0

    return image


# Load images from folder
def load_images(folder_path):

    images = []

    for filename in os.listdir(folder_path):

        image_path = os.path.join(folder_path, filename)

        image = preprocess_image(image_path)

        images.append(image)

    images = np.array(images)

    # Add channel dimension
    images = np.expand_dims(images, axis=-1)

    return images


# Main function
if __name__ == "__main__":

    image_folder = "dataset/segmentation/images"

    mask_folder = "dataset/segmentation/combined_masks"

    images = load_images(image_folder)

    masks = load_images(mask_folder)

    print("Images Shape:", images.shape)

    print("Masks Shape:", masks.shape)