import jax
import optax
import pickle
import datetime
import typing as t
import jax.numpy as jnp
import tensorflow as tf

from tqdm import tqdm
from jax import random
from pathlib import Path
from einops import repeat
from collections import defaultdict
from flax.core.frozen_dict import FrozenDict

from u2net.model import U2Net
from u2net.dataset import duts_dataset


def bce_loss(preds: jnp.ndarray, labels: jnp.ndarray) -> float:
    EPS = 1e-8
    preds = jnp.clip(preds, EPS, 1 - EPS)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds + EPS))
    return loss


def main():
    IMAGE_SIZE = 320
    img_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Image")
    label_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Mask")
    date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

    tf.config.set_visible_devices([], "GPU")
    train_writer = tf.summary.create_file_writer(f"logs/{date}/train")
    valid_writer = tf.summary.create_file_writer(f"logs/{date}/valid")

    batch_size = 4
    train_ds, val_ds = duts_dataset(img_dir, label_dir, batch_size)
    sample_train_img, sample_train_lab = next(iter(train_ds))
    sample_val_img, sample_val_lab = next(iter(val_ds))

    epochs = 1
    mid = [64] * 11
    out = 64
    kernel = (3, 3)
    log_every = 5

    x = jnp.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3))
    model = U2Net(mid, out, kernel)
    key = random.PRNGKey(0)
    params = model.init(key, x)

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(
        params: FrozenDict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        weights: jnp.ndarray = jnp.ones(7),
    ) -> jnp.ndarray:
        saliency_maps = model.apply(params, xs)
        ys = repeat(ys, "b h w 1 -> b h w x", x=len(weights))
        losses = bce_loss(saliency_maps, ys)
        total_loss = jnp.mean(weights * losses)

        return total_loss

    @jax.jit
    def update(
        params: FrozenDict, opt_state: optax.OptState, xs: jnp.ndarray, ys: jnp.ndarray
    ) -> t.Tuple[FrozenDict, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    @jax.jit
    def mean_absolute_error(params: FrozenDict, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
        saliency_maps = model.apply(params, xs)[..., [0]]
        return (saliency_maps - ys).mean()

    with train_writer.as_default():
        tf.summary.image("images", sample_train_img, step=0, max_outputs=8)
        tf.summary.image("labels", sample_train_lab, step=0, max_outputs=8)

    with valid_writer.as_default():
        tf.summary.image("images", sample_val_img, step=0, max_outputs=8)
        tf.summary.image("labels", sample_val_lab, step=0, max_outputs=8)

    train_jax_img = jnp.asarray(sample_train_img)
    val_jax_img = jnp.asarray(sample_val_img)

    for e in range(epochs):
        train_metrics = defaultdict(lambda: tf.keras.metrics.Mean())
        val_metrics = defaultdict(lambda: tf.keras.metrics.Mean())
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=f"Train Epoch {e}")
        val_bar = tqdm(val_ds, total=len(val_ds), ncols=0, desc=f"Valid Epoch {e}")

        for xs, ys in train_bar:
            xs = jnp.asarray(xs)
            ys = jnp.asarray(ys)

            params, opt_state, loss = update(params, opt_state, xs, ys)
            train_metrics["loss"].update_state(loss)
            mae = mean_absolute_error(params, xs, ys)
            train_metrics["mae"].update_state(mae)

            train_bar.set_postfix(**{k: v.result().numpy() for k, v in train_metrics.items()})

        for xs, ys in val_bar:
            xs = jnp.asarray(xs)
            ys = jnp.asarray(ys)

            loss = loss_fn(params, xs, ys)
            val_metrics["loss"].update_state(loss)
            mae = mean_absolute_error(params, xs, ys)
            val_metrics["mae"].update_state(mae)

            val_bar.set_postfix(**{k: v.result().numpy() for k, v in val_metrics.items()})

        with train_writer.as_default():
            tf.summary.scalar("loss", train_metrics["loss"].result().numpy(), step=e)
            tf.summary.scalar("mae", train_metrics["mae"].result().numpy(), step=e)
        with valid_writer.as_default():
            tf.summary.scalar("loss", val_metrics["loss"].result().numpy(), step=e)
            tf.summary.scalar("mae", val_metrics["mae"].result().numpy(), step=e)

        if e % log_every == 0:
            pickle.dump(params, open("weights.pkl", "wb"))
            pickle.dump(opt_state, open("optimizer.pkl", "wb"))

            with train_writer.as_default():
                pred_train = model.apply(params, train_jax_img)[..., [0]]
                tf.summary.image("predictions", pred_train, step=e, max_outputs=8)

            with valid_writer.as_default():
                pred_val = model.apply(params, val_jax_img)[..., [0]]
                tf.summary.image("predictions", pred_val, step=e, max_outputs=8)

        train_writer.flush()
        valid_writer.flush()


if __name__ == "__main__":
    main()
