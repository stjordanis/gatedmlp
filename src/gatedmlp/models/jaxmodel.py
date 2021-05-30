import flax.linen as nn
import jax.numpy as jnp


class SpatialGatingUnit(nn.Module):
    def __init__(self, seq_len, **kwargs):
        super().__init__()

        self.norm = nn.LayerNorm(name="norm")
        self.proj = nn.Dense(features=seq_len, name="proj")

    @nn.compact
    def __call__(self, x):
        u, v = jnp.split(x, 2, axis=-1)
        v = self.norm(v)
        v = jnp.transpose(v, axes=[0, 2, 1])
        v = self.proj(v)
        v = jnp.transpose(v, axes=[0, 2, 1])
        return u * v
