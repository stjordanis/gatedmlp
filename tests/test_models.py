import flax.linen as nn
import tensorflow as tf
import torch

from src.gatedmlp.models import jaxmodel, tfmodel, torchmodel


def test_torch_instance():
    """
    Instance checks for PyTorch Models and Modules
    """
    unit = torchmodel.SpatialGatingUnit(d_ffn=30, seq_len=100)
    block = torchmodel.gMLPBlock(d_model=500, d_ffn=30, seq_len=100)
    assert isinstance(unit, torch.nn.Module)
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
