import tensorflow as tf


class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, seq_len: int):
        super(SpatialGatingUnit, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization()
        self.proj = tf.keras.layers.Dense(
            input_shape=(self.seq_len,), units=self.seq_len
        )

    def call(self, x):
        u, v = tf.split(x, num_or_size_splits=2, axis=-1, name="split")
        v = self.norm(v)
        v = tf.transpose(v, perm=[0, 2, 1])
        v = self.proj(v)
        v = tf.transpose(v, perm=[0, 2, 1])
        return u * v
