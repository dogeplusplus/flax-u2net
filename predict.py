import jax
import pickle
import numpy as np
import streamlit as st
import jax.numpy as jnp

from PIL import Image
from einops import repeat

from u2net.model import U2Net
from u2net.dataset import normalize_image


st.set_page_config(layout="wide")


@st.cache()
def load_params():
    return pickle.load(open("weights.pkl", "rb"))


def main():
    IMAGE_SIZE = 320
    mid = [64] * 11
    out = 64
    kernel = (3, 3)
    model = U2Net(mid, out, kernel)
    params = load_params()

    def predict(image):
        image = np.array(image)
        image = normalize_image(image, -1, 1)
        H, W, C = image.shape
        image = jnp.asarray(image)
        image = jax.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE, 3],
                                 method="bilinear")
        image = repeat(image, "h w c -> 1 h w c")
        saliency_maps = model.apply(params, image)
        prediction = saliency_maps[0]

        pred_resized = jax.image.resize(prediction, [H, W, C],
                                        method="bilinear")
        return np.array(pred_resized)

    allowed_extensions = ["png", "jpg", "jpeg"]
    st.title("Salient Object Detection")
    preamble = (
        "Salient Object Detection with `U-2-Net` implemented with `Flax` and `JAX`."
        "Original implementation can be found at https://github.com/xuebinqin/U-2-Net, "
        "and a link to the paper can be found at [U2-Net](https://arxiv.org/pdf/2005.09007.pdf)."
    )
    st.write(preamble)

    image_upload = st.sidebar.file_uploader("Upload Image", type=allowed_extensions)

    if image_upload is not None:
        left, right = st.columns(2)

        left.subheader("Image")
        right.subheader("Prediction")

        image = Image.open(image_upload)
        pred = predict(image)

        left.image(image)
        right.image(pred)


if __name__ == "__main__":
    main()