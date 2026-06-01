import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# Build CNN Model
def build_cnn(input_shape=(256, 256, 1), num_classes=4):

    model = Sequential()

    # Layer 1
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # Layer 2
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation='relu'
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # Layer 3
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu'
        )
    )

    model.add(MaxPooling2D((2, 2)))

    # Flatten
    model.add(Flatten())

    # Dense
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    # Output
    model.add(Dense(num_classes, activation='softmax'))

    return model


# Test
if __name__ == "__main__":

    model = build_cnn()

    model.summary()