import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

from ..layers import FixedEmbedding
from ..data.qgen_ca_q import CA_Qcfg


HIDDEN_SIZE = 32
LATENT_DIM = 32


class CAZ_Q_Encoder(Model):
    def __init__(self, word_emb_mat, **kwargs):
        super(CAZ_Q_Encoder, self).__init__(**kwargs)
        self.word_emb_mat = word_emb_mat
        self.context_embedding = FixedEmbedding(
            self.word_emb_mat,
            CA_Qcfg.CSEQ_LEN, name="Context-Embedding")
        self.bigru_1 = layers.Bidirectional(layers.GRU(
            HIDDEN_SIZE//2, return_sequences=True, return_state=True,
            name="Context-Encoder-1"))
        # self.bigru_2 = layers.Bidirectional(layers.GRU(
        #         HIDDEN_SIZE/2, return_sequences=True,
        #         return_state=True, name="Context-Encoder-2"))
        self.dense_reparam = layers.Dense(LATENT_DIM*2)
        self.dense_enc = layers.Dense(HIDDEN_SIZE, activation='tanh')

    def call(self, cidx, aidx, enc_hidden):
        # cidx (batch, 250)
        # aidx (batch, 250)
        # enc_hidden (batch, HIDDEN_SIZE)

        cemb = self.context_embedding(cidx)  # (batch, 250, 300)

        # (batch, 250, HIDDEN_SIZE),
        # (batch, HIDDEN_SIZE/2),
        # (batch, HIDDEN_SIZE/2)
        cop, cfinalfenc, cfinalbenc = self.bigru_1(
            cemb, initial_state=enc_hidden)

        # (batch, HIDDEN_SIZE)
        cfinalenc = tf.concat([cfinalfenc, cfinalbenc], -1)

        # (batch, 250, 1)
        aidx = tf.cast(tf.expand_dims(aidx, -1), tf.float32)
        # (batch, HIDDEN_SIZE)
        aenc = tf.reduce_mean(tf.multiply(cop, aidx), 1)

        meanlogvar = tf.concat([cfinalenc, aenc], -1)  # (batch, LATENT_DIM*2)

        # (batch, LATENT_DIM), (batch, LATENT_DIM)
        meanlogvar = self.dense_reparam(meanlogvar)

        mean, logvar = tf.split(
            meanlogvar, num_or_size_splits=2, axis=1)

        # reparameterization
        eps = tf.random.normal(shape=mean.shape)  # (batch, LATENT_DIM)
        z = eps * tf.exp(logvar * 0.5) + mean  # (batch, LATENT_DIM)

        # input (batch, HIDDEN_SIZE+HIDDEN_SIZE+LATENT_DIM)
        # output (batch, HIDDEN_SIZE)
        s0 = self.dense_enc(tf.concat([cfinalenc, aenc, z], -1))

        # (batch, HIDDEN_SIZE),
        # (batch, 250, HIDDEN_SIZE),
        # (batch, LATENT_DIM),
        # (batch, LATENT_DIM),
        # (batch, LATENT_DIM)
        return s0, cop, mean, logvar, z

    def initialize_hidden_size(self, batch_sz):
        return [
            tf.zeros((batch_sz, HIDDEN_SIZE//2)),
            tf.zeros((batch_sz, HIDDEN_SIZE//2)),
        ]
