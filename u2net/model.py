import jax
import typing as t
import jax.numpy as jnp
import flax.linen as nn

from dataclasses import field


IMAGE_SIZE = 128


def upsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    B, H, W, C = x.shape
    x = jax.image.resize(x, (B, H * factor, W * factor, C), method="bilinear")
    return x


class ConvLNRelu(nn.Module):
    out: int
    kernel: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel, kernel_dilation=self.dilation)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        return x


class RSUBlock(nn.Module):
    levels: int
    out: int
    kernel: int
    mid: int

    @nn.compact
    def __call__(self, x):
        down_levels = [
            ConvLNRelu(self.mid, self.kernel)
            for _ in range(self.levels - 1)
        ]

        up_levels = [
            ConvLNRelu(self.mid, self.kernel)
            for _ in range(self.levels - 1)
        ]

        top_left = ConvLNRelu(self.out, self.kernel)(x)

        x = top_left
        down_stack = []
        for layer in down_levels:
            x = layer(x)
            down_stack.insert(0, x)
            x = nn.max_pool(x, (2, 2), (2, 2))

        # Insert another convolution without the pooling at the bottom
        down_stack.insert(0, ConvLNRelu(self.mid, self.kernel)(x))

        x = ConvLNRelu(self.mid, self.kernel, 2)(x)

        for down, layer in zip(down_stack, up_levels):
            x = jnp.concatenate([down, x], axis=-1)
            x = layer(x)
            x = upsample(x, 2)

        # Final convolution at the top right before concatenation
        x = ConvLNRelu(self.out, self.kernel)(x)
        out = top_left + x

        return out


class DilationRSUBlock(nn.Module):
    out: int
    kernel: int
    mid: int

    @nn.compact
    def __call__(self, x):
        top_left = ConvLNRelu(self.out, self.kernel)(x)

        x = top_left
        d1 = ConvLNRelu(self.mid, self.kernel)(x)
        d2 = ConvLNRelu(self.mid, self.kernel)(d1)
        d3 = ConvLNRelu(self.mid, self.kernel, dilation=2)(d2)
        d4 = ConvLNRelu(self.mid, self.kernel, dilation=4)(d3)

        b = ConvLNRelu(self.mid, self.kernel, dilation=8)(d4)

        u4 = ConvLNRelu(self.mid, self.kernel, dilation=4)(
            jnp.concatenate([d4, b], axis=-1)
        )
        u3 = ConvLNRelu(self.mid, self.kernel, dilation=4)(
            jnp.concatenate([d3, u4], axis=-1)
        )
        u2 = ConvLNRelu(self.mid, self.kernel, dilation=2)(
            jnp.concatenate([d2, u3], axis=-1)
        )
        u1 = ConvLNRelu(self.out, self.kernel)(
            jnp.concatenate([d1, u2], axis=-1)
        )

        out = top_left + u1
        return out


class SideSaliency(nn.Module):
    target_shape: t.Tuple[int, int, int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, (3, 3))(x)
        x = jax.image.resize(x, self.target_shape, method="bilinear")
        x = jax.nn.sigmoid(x)

        return x


class U2Net(nn.Module):
    mid: t.List[int] = field(default_factory=lambda: [16] * 11)
    out: int = 64
    kernel: t.Tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, x):
        B, H, W, _ = x.shape

        en1 = RSUBlock(7, self.out, self.kernel, self.mid[0])(x)
        x = nn.max_pool(en1, (2, 2), (2, 2))

        en2 = RSUBlock(6, self.out, self.kernel, self.mid[1])(x)
        x = nn.max_pool(en2, (2, 2), (2, 2))

        en3 = RSUBlock(5, self.out, self.kernel, self.mid[2])(x)
        x = nn.max_pool(en3, (2, 2), (2, 2))

        en4 = RSUBlock(4, self.out, self.kernel, self.mid[3])(x)
        x = nn.max_pool(en4, (2, 2), (2, 2))

        en5 = DilationRSUBlock(self.out, self.kernel, self.mid[4])(x)

        en6 = DilationRSUBlock(self.out, self.kernel, self.mid[5])(x)
        sup6 = SideSaliency((B, H, W, 1))(en6)

        x = jnp.concatenate([en5, en6], axis=-1)
        x = upsample(x, 2)
        de5 = DilationRSUBlock(self.out, self.kernel, self.mid[6])(x)
        sup5 = SideSaliency((B, H, W, 1))(de5)

        x = jnp.concatenate([de5, en4], axis=-1)
        x = upsample(x, 2)
        de4 = RSUBlock(4, self.out, self.kernel, self.mid[7])(x)
        sup4 = SideSaliency((B, H, W, 1))(de4)

        x = jnp.concatenate([de4, en3], axis=-1)
        x = upsample(x, 2)
        de3 = RSUBlock(5, self.out, self.kernel, self.mid[8])(x)
        sup3 = SideSaliency((B, H, W, 1))(de3)

        x = jnp.concatenate([de3, en2], axis=-1)
        x = upsample(x, 2)
        de2 = RSUBlock(6, self.out, self.kernel, self.mid[9])(x)
        sup2 = SideSaliency((B, H, W, 1))(de2)

        x = jnp.concatenate([de2, en1], axis=-1)
        x = upsample(x, 2)
        de1 = RSUBlock(7, self.out, self.kernel, self.mid[10])(x)
        sup1 = SideSaliency((B, H, W, 1))(de1)

        fused = jnp.concatenate([sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)
        fused = nn.Conv(1, (1, 1))(fused)
        out = jax.nn.sigmoid(fused)

        return jnp.concatenate([out, sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)
