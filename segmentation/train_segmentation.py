import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from attention_unet import build_unet


# =====================================
# SETTINGS
# =====================================

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 75


# =====================================
# DICE LOSS
# =====================================

def dice_loss(y_true, y_pred):

    smooth = 1e-6

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true_f)
        + tf.reduce_sum(y_pred_f)
        + smooth
    )

    return 1 - dice


# =====================================
# LOAD DATASET
# =====================================

def load_dataset(
    image_folder,
    mask_folder,
    image_files,
    mask_files
):

    images = []
    masks = []

    for img_file, mask_file in zip(
        image_files,
        mask_files
    ):

        img_path = os.path.join(
            image_folder,
            img_file
        )

        mask_path = os.path.join(
            mask_folder,
            mask_file
        )

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        mask = mask.astype(
            np.float32
        ) / 255.0

        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    images = np.expand_dims(
        images,
        axis=-1
    )

    masks = np.expand_dims(
        masks,
        axis=-1
    )

    return images, masks


# =====================================
# MATCH IMAGE-MASK PAIRS
# =====================================

image_folder = "dataset/segmentation/images"
mask_folder = "dataset/segmentation/combined_masks"

image_files = sorted(
    os.listdir(image_folder)
)

mask_files = sorted(
    os.listdir(mask_folder)
)

mask_map = {}

for mask in mask_files:

    image_name = mask.replace(
        "_mask",
        ""
    )

    mask_map[image_name] = mask

matched_images = []
matched_masks = []

for image in image_files:

    if image in mask_map:

        matched_images.append(
            image
        )

        matched_masks.append(
            mask_map[image]
        )

print(
    f"Matched Pairs: {len(matched_images)}"
)


# =====================================
# LOAD MATCHED DATASET
# =====================================

X, Y = load_dataset(
    image_folder,
    mask_folder,
    matched_images,
    matched_masks
)

print(
    "Images Shape:",
    X.shape
)

print(
    "Masks Shape:",
    Y.shape
)


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# =====================================
# BUILD MODEL
# =====================================

model = build_unet()

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss=dice_loss,
    metrics=["accuracy"]
)

model.summary()


# =====================================
# SAVE FOLDER
# =====================================

os.makedirs(
    "saved_models",
    exist_ok=True
)


# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "saved_models/segmentation_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)


# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_test,
        Y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# =====================================
# SAVE FINAL MODEL
# =====================================

model.save(
    "saved_models/segmentation_model_final.h5"
)

print("\n✅ TRAINING COMPLETED!")
print("✅ BEST MODEL SAVED!")