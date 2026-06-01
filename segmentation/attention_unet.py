from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# Convolution Block
def conv_block(x, filters):

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# Encoder Block
def encoder_block(x, filters):

    f = conv_block(x, filters)

    p = MaxPooling2D((2, 2))(f)

    return f, p


# Decoder Block
def decoder_block(x, skip, filters):

    x = UpSampling2D((2, 2))(x)

    x = Concatenate()([x, skip])

    x = conv_block(x, filters)

    return x


# Build U-Net
def build_unet(input_shape=(256, 256, 1)):

    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)

    s2, p2 = encoder_block(p1, 128)

    s3, p3 = encoder_block(p2, 256)

    # Bottleneck
    b1 = conv_block(p3, 512)

    # Decoder
    d1 = decoder_block(b1, s3, 256)

    d2 = decoder_block(d1, s2, 128)

    d3 = decoder_block(d2, s1, 64)

    # Output
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    model = Model(inputs, outputs)

    return model


# Test model
if __name__ == "__main__":

    model = build_unet()

    model.summary()