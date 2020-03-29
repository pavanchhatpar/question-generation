import tensorflow as tf

from .fixed_embedding import FixedEmbedding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, qword_emb_mat, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.question_embedding = FixedEmbedding(
            qword_emb_mat, config.MAX_QLEN, name="Question-Embedding")
        self.gru = tf.keras.layers.GRU(
            32, return_sequences=True, return_state=True, name="GRU-Decoder")
        self.decode_attention = tf.keras.layers.AdditiveAttention(
            causal=True, name="Decoder-Attention")
        self.dense_dec = tf.keras.layers.Dense(
            config.QVOCAB_SIZE, name="Decoder-Dense")

    def call(self, inputs):
        q_idx, s0, hd = inputs
        q_embs = self.question_embedding(q_idx)
        q_int_dec, st = self.gru(q_embs, initial_state=s0)
        # inputs = [query:Tensor, value:Tensor]
        q_dec = self.decode_attention(inputs=[q_int_dec, hd])
        y = self.dense_dec(q_dec)
        return y, st
