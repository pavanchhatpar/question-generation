import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

from ..layers import FixedEmbedding
from ..data.qgen_ca_q import CA_Qcfg


class CAZ_Q_Encoder(Model):
    def __init__(self, word_emb_mat, **kwargs):
        super(CAZ_Q_Encoder, self).__init__(**kwargs)
        self.word_emb_mat = word_emb_mat
        self.context_embedding = FixedEmbedding(
            self.word_emb_mat,
            CA_Qcfg.CSEQ_LEN, name="Context-Embedding")
        self.bigru_1 = layers.Bidirectional(layers.GRU(
            16, return_sequences=True, return_state=True,
            name="Context-Encoder-1"))
        # self.bigru_2 = layers.Bidirectional(layers.GRU(
        #         16, return_sequences=True,
        #         return_state=True, name="Context-Encoder-2"))
        self.dense_reparam = layers.Dense(64)
        self.dense_enc = layers.Dense(32, activation='tanh')

    def call(self, cidx, aidx, enc_hidden):
        # cidx (batch, 250)
        # aidx (batch, 250)
        # enc_hidden (batch, 32)

        cemb = self.context_embedding(cidx)  # (batch, 250, 300)

        # (batch, 250, 32), (batch, 16), (batch, 16)
        cop, cfinalfenc, cfinalbenc = self.bigru_1(cemb)

        # (batch, 32)
        cfinalenc = tf.concat([cfinalfenc, cfinalbenc], -1)

        aidx = tf.cast(tf.expand_dims(aidx, -1), tf.float32)  # (batch, 250, 1)
        aenc = tf.reduce_mean(tf.multiply(cop, aidx), 1)  # (batch, 32)

        meanlogvar = tf.concat([cfinalenc, aenc], -1)  # (batch, 64)

        # (batch, 32), (batch, 32)
        mean, logvar = tf.split(
            self.dense_reparam(meanlogvar), num_or_size_splits=2, axis=1)

        # reparameterization
        eps = tf.random.normal(shape=mean.shape)  # (batch, 32)
        z = eps * tf.exp(logvar * 0.5) + mean  # (batch, 32)

        # input
        # (batch, 96)
        s0 = self.dense_enc(tf.concat([cfinalenc, aenc, z], -1))  # (batch, 32)

        # (batch, 32), (batch, 250, 32), (batch, 32), (batch, 32), (batch, 32)
        return s0, cop, mean, logvar, z

    def initialize_hidden_size(self, batch_sz):
        return tf.zeros((batch_sz, 32))
