import math
import typing as t
import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path


def normalize_image(img: tf.Tensor, a=-1, b=1) -> tf.Tensor:
    lower = tf.reduce_min(img)
    upper = tf.reduce_max(img)
    img = (img - lower) / (upper - lower)
    img = a + (b - a) * img
    return img


def parse_image(filename: str, channels=3) -> tf.Tensor:
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [320, 320])
    return image


def augmentation(images: tf.Tensor, labels: tf.Tensor) -> t.Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform([]) > 0.5:
        images = tf.image.flip_left_right(images)
        labels = tf.image.flip_left_right(labels)

    if tf.random.uniform([]) > 0.5:
        delta = tf.random.uniform([], maxval=0.2)
        images = tf.image.adjust_brightness(images, delta)
        labels = tf.image.adjust_brightness(labels, delta)

    if tf.random.uniform([]) > 0.5:
        angle = tf.random.uniform([], 0, math.pi/6)
        images = tfa.image.rotate(images, angle)
        labels = tfa.image.rotate(labels, angle)

    return images, labels


def duts_dataset(
    img_dir: Path,
    label_dir: Path,
    batch_size: int,
    val_ratio: float = 0.2,
    shuffle_buffer: int = 8,
):
    images = tf.data.Dataset.list_files(str(img_dir / "*"), shuffle=False)
    labels = images.map(
        lambda x: tf.strings.regex_replace(x, str(img_dir), str(label_dir))
    )

    images = images.map(parse_image)
    images = images.map(lambda x: normalize_image(x, -1, 1))

    labels = labels.map(lambda x: tf.strings.regex_replace(x, ".jpg", ".png"))
    labels = labels.map(lambda x: parse_image(x, 1))
    labels = labels.map(lambda x: normalize_image(x, 0, 1))

    ds = tf.data.Dataset.zip((images, labels))
    size = len(ds)
    val_size = int(size * val_ratio)

    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)

    train_ds = train_ds.map(augmentation)

    train_ds = (
        train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds
