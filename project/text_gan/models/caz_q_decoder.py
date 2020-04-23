import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras import Model

from ..layers import FixedEmbedding, AdditiveAttention
from ..data.qgen_ca_q import CA_Qcfg


HIDDEN_SIZE = 32


class CAZ_Q_Decoder(Model):
    def __init__(self, qword_emb_mat, **kwargs):
        super(CAZ_Q_Decoder, self).__init__(**kwargs)
        self.question_embedding = FixedEmbedding(
            qword_emb_mat, CA_Qcfg.QSEQ_LEN, name="Question-Embedding")
        self.gru = layers.GRU(
            HIDDEN_SIZE, return_sequences=True, return_state=True, name="GRU-Decoder")
        self.decode_attention = AdditiveAttention(
            causal=True, name="Decoder-Attention", use_scale=False)
        self.dense_dec = layers.Dense(
            CA_Qcfg.QVOCAB_SIZE, name="Decoder-Dense")

    def call(self, q_idx, s0, hd):
        # q_idx (batch, 1)
        # s0 (batch, HIDDEN_SIZE)
        # hd (batch, 250, HIDDEN_SIZE)

        s0 = tf.expand_dims(s0, 1)  # (batch, 1, HIDDEN_SIZE)

        # inputs = [query: (batch, 1, HIDDEN_SIZE),
        # value: (batch, 250, HIDDEN_SIZE)]
        # (batch, 1, HIDDEN_SIZE), (batch, 1, 250)
        c0, attn_weights = self.decode_attention(inputs=[s0, hd])

        q_embs = self.question_embedding(q_idx)  # (batch, 1, 300)

        gruin = tf.concat([q_embs, c0], -1)  # (batch, 1, 300+HIDDEN_SIZE)

        # (batch, 1, HIDDEN_SIZE), (batch, HIDDEN_SIZE)
        q_dec, st = self.gru(gruin)

        y = self.dense_dec(q_dec)   # (batch, 1, 5000)
        return y, st  # (batch, 1, 5000), (batch, HIDDEN_SIZE)
