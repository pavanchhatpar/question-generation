import tensorflow.keras.layers as layers
import tensorflow as tf

from .fixed_embedding import FixedEmbedding
from ..data.qgen_ca_q import CA_Qcfg


class CA_Q_Decoder(layers.Layer):
    def __init__(self, qword_emb_mat, **kwargs):
        super(CA_Q_Decoder, self).__init__(**kwargs)
        self.question_embedding = FixedEmbedding(
            qword_emb_mat, CA_Qcfg.QSEQ_LEN, name="Question-Embedding")
        self.gru = layers.GRU(
            32, return_sequences=True, return_state=True, name="GRU-Decoder")
        self.decode_attention = layers.AdditiveAttention(
            causal=True, name="Decoder-Attention", use_scale=False)
        self.dense_dec = layers.Dense(
            CA_Qcfg.QVOCAB_SIZE, name="Decoder-Dense", activation='softmax')

    def call(self, inputs):
        q_idx, s0, hd = inputs
        q_embs = self.question_embedding(q_idx)
        q_int_dec, st = self.gru(q_embs, initial_state=s0)
        # inputs = [query:Tensor, value:Tensor]
        q_dec = self.decode_attention(inputs=[q_int_dec, hd])
        q_dec = tf.multiply(q_dec, tf.expand_dims(st, 1))
        y = self.dense_dec(q_dec)
        return y, st
