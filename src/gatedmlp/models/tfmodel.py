import tensorflow as tf
import tensorflow_addons as tfa


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


class gMLPBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        # survival_prob
    ):
        super(gMLPBlock, self).__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn
        self.seq_len = seq_len
        # self.prob = survival_prob

    def build(self, input_shape):

        self.norm = tf.keras.layers.LayerNormalization()
        self.proj_1 = tf.keras.layers.Dense(
            input_shape=(self.d_model,), units=self.d_ffn
        )
        self.activation = tfa.layers.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(self.seq_len)
        self.proj_2 = tf.keras.layers.Dense(
            input_shape=(self.d_model // 2,), units=self.d_model
        )
