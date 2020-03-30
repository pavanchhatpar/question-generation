import tensorflow as tf
import tensorflow.keras.layers as layers

from .fixed_embedding import FixedEmbedding
from ..data.qgen_ca_q import CA_Qcfg


class CA_Q_Encoder(layers.Layer):
    def __init__(self, word_emb_mat, **kwargs):
        super(CA_Q_Encoder, self).__init__(**kwargs)
        self.word_emb_mat = word_emb_mat

    def build(self, input_shape):
        self.context_embedding = FixedEmbedding(
            self.word_emb_mat,
            CA_Qcfg.CSEQ_LEN, name="Context-Embedding")
        self.bigru_1 = layers.Bidirectional(layers.GRU(
            16, return_sequences=True, name="Context-Encoder-1"))
        self.bigru_2 = layers.Bidirectional(layers.GRU(
                16, return_sequences=True,
                return_state=True, name="Context-Encoder-2"))
        self.dense_enc = layers.Dense(32, activation='tanh')

    def call(self, inputs):
        cidx, aidx = inputs
        cemb = self.context_embedding(cidx)

        cintenc = self.bigru_1(cemb)
        cop, cfinalfenc, cfinalbenc = self.bigru_2(cintenc)
        cfinalenc = tf.concat([cfinalfenc, cfinalbenc], -1)

        aidx = tf.cast(tf.expand_dims(aidx, -1), tf.float32)
        aenc = tf.reduce_mean(tf.multiply(cop, aidx), 1)

        op = tf.concat([cfinalenc, aenc], -1)
        op = self.dense_enc(op)

        return op, cop
