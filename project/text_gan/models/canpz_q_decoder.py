import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras import Model

from ..layers import FixedEmbedding, AdditiveAttention
from ..config import cfg


class CANPZ_Q_Decoder(Model):
    def __init__(self, qembs, **kwargs):
        super(CANPZ_Q_Decoder, self).__init__(**kwargs)

        # embedding layers
        self.token_emb_layer = FixedEmbedding(qembs.get_matrix(), cfg.QSEQ_LEN)

        # gru
        self.gru = layers.GRU(
            cfg.HIDDEN_DIM, return_sequences=True, return_state=True)

        # attention
        self.dec_attn = AdditiveAttention()

        # dense layers
        self.dec = layers.Dense(len(qembs.vocab))

    def call(self, qidx, s0, hd, training=None):
        s0a = s0

        # (, h) => (, 1, h)
        s0 = tf.expand_dims(s0, 1)

        # (, 1, h), (, 1, 250)
        c0, attn_weights = self.dec_attn(inputs=[s0, hd])

        # (, 1) => (, 1, 300)
        tokenemb = self.token_emb_layer(qidx)

        # (, 1, 300), (, 1, h) => (, 1, h+300)
        qemb = tf.concat([tokenemb, c0], -1)

        # (, 1, h), (, h)
        qdec, s1 = self.gru(qemb, initial_state=s0a)

        # (, 1, h) => (, 1, v)
        y = self.dec(qdec)

        return y, s1, attn_weights
