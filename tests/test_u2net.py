import jax.numpy as jnp

from jax import random

from u2net.model import (
    ConvLNRelu,
    RSUBlock,
    DilationRSUBlock,
    U2Net,
    SideSaliency,
    upsample,
)


def test_conv_block():
    out = 5
    kernel = (3, 3)

    x = jnp.ones((4, 128, 128, 3))
    layer = ConvLNRelu(out, kernel)
    key = random.PRNGKey(0)
    params = layer.init(key, x)

    y, mutated_vars = layer.apply(params, x)

    assert y.shape == (4, 128, 128, out)
    assert "batch_stats" in mutated_vars.keys()


def test_rsu_block():
    levels = 2
    out = 5
    kernel = (3, 3)
    mid = 16

    x = jnp.ones((4, 128, 128, 3))
    block = RSUBlock(levels, out, kernel, mid)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x)

    assert y.shape == (4, 128, 128, out)


def test_dilation_rsu_block():
    out = 6
    kernel = (3, 3)
    mid = 16

    x = jnp.ones((4, 32, 32, 3))
    block = DilationRSUBlock(out, kernel, mid)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x)

    assert y.shape == (4, 32, 32, out)


def test_upsample():
    x = jnp.ones((2, 32, 32, 3))
    y = upsample(x, 2)

    assert y.shape == (2, 64, 64, 3)


def test_u2_net():
    mid = 16
    out = 64
    kernel = (3, 3)

    x = jnp.ones((4, 256, 256, 3))
    model = U2Net(mid, out, kernel)
    key = random.PRNGKey(0)
    params = model.init(key, x)

    saliency_maps, _ = model.apply(params, x)
    assert saliency_maps.shape == (4, 256, 256, 7)
    assert jnp.max(saliency_maps) <= 1
    assert jnp.min(saliency_maps) >= 0


def test_saliency_map():
    target_shape = (3, 64, 64, 3)
    x = jnp.ones((3, 8, 8, 3))

    layer = SideSaliency(target_shape)
    key = random.PRNGKey(0)
    params = layer.init(key, x)
    saliency_map = layer.apply(params, x)
    assert saliency_map.shape == target_shape
    assert jnp.max(saliency_map) <= 1
    assert jnp.min(saliency_map) >= 0
