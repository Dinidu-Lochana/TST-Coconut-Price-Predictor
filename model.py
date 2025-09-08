import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# -------------------------------
# Positional Encoding Layer
# -------------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # apply sin to even indices, cos to odd
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


# -------------------------------
# Transformer Encoder Block
# -------------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1,trainable=True, **kwargs):
        
        super().__init__(trainable=trainable, **kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.att(x, x, x, attention_mask=mask)  # Self attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)               # Residual + Norm

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)             # Residual + Norm


# -------------------------------
# Time Series Transformer Model
# -------------------------------
class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, input_seq_len=20, input_dim=21, output_seq_len=12,
                 d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input embedding
        self.embedding = layers.Dense(d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_len=input_seq_len, d_model=d_model)

        # Stacked transformer encoder
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, ff_dim, rate=dropout)
            for _ in range(num_layers)
        ]

        # Final Dense projection to forecast horizon
        self.flatten = layers.Flatten()
        self.fc_out = layers.Dense(output_seq_len)

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)

        x = self.flatten(x)
        return self.fc_out(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_seq_len": self.input_seq_len,
            "input_dim": self.input_dim,
            "output_seq_len": self.output_seq_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
