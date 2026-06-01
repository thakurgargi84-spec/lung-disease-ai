import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="conv2d_2"
):

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        class_idx = tf.argmax(
            predictions[0]
        )

        loss = predictions[
            :,
            class_idx
        ]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        np.max(heatmap) + 1e-8
    )

    return heatmap