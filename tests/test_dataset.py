import pytest
import tensorflow as tf

from u2net.dataset import normalize_image, augmentation


@pytest.fixture
def cpu_only():
    tf.config.set_visible_devices([], 'GPU')


@pytest.mark.usefixtures("cpu_only")
def test_normalize():
    image = tf.random.uniform([32, 32, 3], 0, 255)
    norm = normalize_image(image, -1, 1)

    assert tf.reduce_max(norm) <= 1
    assert tf.reduce_min(norm) >= -1


@pytest.mark.usefixtures("cpu_only")
def test_augmentation():
    image = tf.random.uniform([32, 32, 3], 0, 255)
    label = tf.random.uniform([32, 32, 1], 0, 1)

    aug_image, aug_label = augmentation(image, label)
    assert aug_image.shape == image.shape
    assert aug_label.shape == label.shape
