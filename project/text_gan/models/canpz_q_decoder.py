import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from typing import Dict, Any

from ..layers import FixedEmbedding, AdditiveAttention, FixedDense
from ..config import cfg
from ..vocab import Vocab


class CANPZ_Q_Decoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 **kwargs: Dict[str, Any]):
        super(CANPZ_Q_Decoder, self).__init__(**kwargs)

        emb_mat = vocab.get_embedding_matrix("target")
        # embedding layers
        with tf.device("/cpu:0"):
            self.token_emb_layer = FixedEmbedding(
                emb_mat, cfg.QSEQ_LEN)

        # gru
        self.gru = layers.GRU(
            cfg.HIDDEN_DIM, return_sequences=True, return_state=True)

        # attention
        self.dec_attn = AdditiveAttention()

        # dense layers
        with tf.device("/cpu:0"):
            self.dec = FixedDense(
                emb_mat.shape[0],
                weights=[emb_mat.T, np.zeros(emb_mat.shape[0])])

    def call(self, qidx, s0, hd, source_mask, training=None):
        s0a = s0

        # (, h) => (, 1, h)
        s0 = tf.expand_dims(s0, 1)

        # (, 1, h), (, 1, 250)
        c0, attn_weights = self.dec_attn(
            inputs=[s0, hd], mask=[None, source_mask])

        # (, 1) => (, 1, 300)
        with tf.device("/cpu:0"):
            tokenemb = self.token_emb_layer(qidx)

        # (, 1, 300), (, 1, h) => (, 1, h+300)
        qemb = tf.concat([tokenemb, c0], -1)

        # (, 1, h), (, h)
        qdec, s1 = self.gru(qemb, initial_state=s0a)

        # (, 1, h) => (, 1, v)
        with tf.device("/cpu:0"):
            y = self.dec(qdec)

        return y, s1, attn_weights
