import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras import Model

from ..layers import FixedEmbedding
from ..data.qgen_ca_q import CA_Qcfg


class CAZ_Q_Decoder(Model):
    def __init__(self, qword_emb_mat, **kwargs):
        super(CAZ_Q_Decoder, self).__init__(**kwargs)
        self.question_embedding = FixedEmbedding(
            qword_emb_mat, CA_Qcfg.QSEQ_LEN, name="Question-Embedding")
        self.gru = layers.GRU(
            32, return_sequences=True, return_state=True, name="GRU-Decoder")
        self.decode_attention = layers.AdditiveAttention(
            causal=True, name="Decoder-Attention", use_scale=False)
        self.dense_dec = layers.Dense(
            CA_Qcfg.QVOCAB_SIZE, name="Decoder-Dense")

    def call(self, q_idx, s0, hd):
        # q_idx (batch, 1)
        # s0 (batch, 32)
        # hd (batch, 250, 32)

        s0 = tf.expand_dims(s0, 1)  # (batch, 1, 32)

        # inputs = [query:Tensor, value:Tensor]
        c0 = self.decode_attention(inputs=[s0, hd])  # (batch, 1, 32)

        q_embs = self.question_embedding(q_idx)  # (batch, 1, 300)

        gruin = tf.concat([q_embs, c0], -1)  # (batch, 1, 300+32)

        q_dec, st = self.gru(gruin)  # (batch, 1, 32), (batch, 32)

        y = self.dense_dec(q_dec)  # (batch, 1, 5000)
        return y, st  # (batch, 1, 5000), (batch, 32)
