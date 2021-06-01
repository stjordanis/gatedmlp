import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from jax import random

from src.gatedmlp.models import jaxmodel, tfmodel, torchmodel


def test_torch_instance():
    """
    Instance checks for PyTorch Models and Modules
    """
    unit = torchmodel.SpatialGatingUnit(dim=30, dim_seq=100)
    causal_unit = torchmodel.SpatialGatingUnit(dim=30, dim_seq=100, causal=True)
    block = torchmodel.gMLPBlock(d_model=500, d_ffn=30, seq_len=100)
    assert isinstance(unit, torch.nn.Module)
    assert isinstance(causal_unit, torch.nn.Module)
    assert isinstance(block, torch.nn.Module)


def test_tf_instance():
    """
    Instance checks for Tensorflow Models and Modules
    """
    unit = tfmodel.SpatialGatingUnit(seq_len=100)
    block = tfmodel.gMLPBlock(d_model=500, d_ffn=30, seq_len=100)
    assert isinstance(unit, tf.keras.layers.Layer)
    assert isinstance(block, tf.keras.layers.Layer)


def test_flax_instance():
    """
    Instance checks for Flax Models and Modules
    """
    unit = jaxmodel.SpatialGatingUnit(seq_len=100)
    assert isinstance(unit, nn.Module)


def test_torch_pass():
    """
    Pass checks for PyTorch Models and Modules
    """
    temp = torch.randn([100, 100, 30])
    unit = torchmodel.SpatialGatingUnit(dim=30, dim_seq=100)
    block = torchmodel.gMLPBlock(d_model=30, d_ffn=30, seq_len=100)
    out = unit(temp)
    assert out is not None
    out = block(temp)
    assert out is not None


def test_tf_pass():
    """
    Pass checks for Tensorflow Models and Modules
    """
    temp = np.random.randn(100, 100, 30)
    unit = tfmodel.SpatialGatingUnit(seq_len=100)
    block = tfmodel.gMLPBlock(d_model=300, d_ffn=30, seq_len=100)
    out = unit(temp)
    assert out is not None
    out = block(temp)
    assert out is not None


def test_flax_pass():
    """
    Instance checks for Flax Models and Modules
    """
    unit = jaxmodel.SpatialGatingUnit(seq_len=100)
    variables = unit.init(random.PRNGKey(0), jnp.ones((100, 100, 30)))
    out = unit.apply(variables, jnp.ones((100, 100, 30)))
    assert out is not None
