import jax
import json
import pickle
import typing as t
import numpy as np
import streamlit as st
import jax.numpy as jnp
import tensorflow as tf

from PIL import Image
from pathlib import Path
from einops import repeat

from u2net.model import U2Net
from u2net.dataset import normalize_image
from flax.core.frozen_dict import FrozenDict


st.set_page_config(layout="wide")
tf.config.set_visible_devices([], 'GPU')


@st.cache()
def load_model(training_run: Path) -> t.Tuple[FrozenDict, t.Dict[str, t.Any]]:
    weights_path = training_run / "weights.pkl"
    model_config_path = training_run / "model.json"
    weights = pickle.load(open(weights_path, "rb"))

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    return weights, model_config


def main():
    model_instances = "models"

    runs = list(Path(model_instances).iterdir())
    selected_run = st.sidebar.selectbox("Training Run", runs)

    params, model_config = load_model(selected_run)
    model = U2Net(model_config["mid"], model_config["out"], model_config["kernel"])
    image_size = model_config["image_size"]

    def predict(image):
        image = np.array(image)
        image = normalize_image(image, -1, 1)
        H, W, C = image.shape
        image = jnp.asarray(image)
        image = jax.image.resize(image, [image_size, image_size, 3],
                                 method="bilinear")
        image = repeat(image, "h w c -> 1 h w c")
        saliency_maps = model.apply(params, image)
        prediction = saliency_maps[0]

        pred_resized = jax.image.resize(prediction, [H, W, C],
                                        method="bilinear")
        pred_resized = np.clip(np.asarray(pred_resized), 0, 1)
        return pred_resized

    allowed_extensions = ["png", "jpg", "jpeg"]
    st.title("Salient Object Detection 🔎")
    preamble = (
        "Salient Object Detection with `U-2-Net` implemented with `Flax` and `JAX`."
        "Original implementation can be found at https://github.com/xuebinqin/U-2-Net, "
        "and a link to the paper can be found at [U2-Net](https://arxiv.org/pdf/2005.09007.pdf)."
    )
    st.write(preamble)

    image_upload = st.sidebar.file_uploader("Upload Image", type=allowed_extensions)

    if image_upload is not None:
        left, right = st.columns(2)

        left.subheader("Image 📷")
        right.subheader("Prediction 💻")
        image = Image.open(image_upload)
        pred = predict(image)

        left.image(image)
        right.image(pred)


if __name__ == "__main__":
    main()
