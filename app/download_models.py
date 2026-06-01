import os
import gdown

os.makedirs("saved_models", exist_ok=True)

cnn_model_path = "saved_models/cnn_prediction_model.h5"
seg_model_path = "saved_models/segmentation_model.h5"

if not os.path.exists(cnn_model_path):
    gdown.download(
        id="1ZTxIAu8OgSfinMyneA7s7fLayk7DXOmA",
        output=cnn_model_path,
        quiet=False
    )

if not os.path.exists(seg_model_path):
    gdown.download(
        id="1a-jRs-s52GKcbWcbBTtIjs47GTJK5Pvk",
        output=seg_model_path,
        quiet=False
    )