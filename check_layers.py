from tensorflow.keras.models import load_model

model = load_model(
    "saved_models/cnn_prediction_model.h5"
)

for layer in model.layers:
    print(layer.name)