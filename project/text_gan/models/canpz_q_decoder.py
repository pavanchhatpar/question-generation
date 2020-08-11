import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from typing import Dict, Any
from copynet_tf import Vocab
from copynet_tf.layers import BahdanauAttention

from ..config import cfg


class CANPZ_Q_Decoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 question_emb_layer,
                 question_dec_layer,
                 **kwargs: Dict[str, Any]):
        super(CANPZ_Q_Decoder, self).__init__(**kwargs)

        # emb_mat = vocab.get_embedding_matrix("target")
        # embedding layers
        # with tf.device("cpu:0"):
        self.token_emb_layer = question_emb_layer

        # gru
        self.gru = layers.GRU(
            cfg.HIDDEN_DIM, return_sequences=True, return_state=True)

        # attention
        self.dec_attn = BahdanauAttention(cfg.HIDDEN_DIM)

        # dense layers
        # with tf.device("cpu:0"):
        self.dec = question_dec_layer

    def call(self, qidx, state, training=None):

        if len(qidx.shape) == 1:
            qidx = tf.expand_dims(qidx, 1)

        s0 = state["s0"]
        hd = state["hd"]
        source_mask = state["source_mask"]

        # (, h), (, 250, 1)
        c0, attn_weights = self.dec_attn(s0, hd, source_mask)

        # (, 1, h)
        c0 = tf.expand_dims(c0, 1)

        attn_weights = tf.squeeze(attn_weights, axis=-1)

        # (, 1, 250)
        attn_weights = tf.expand_dims(attn_weights, 1)

        # (, 1) => (, 1, 300)
        with tf.device("cpu:0"):
            tokenemb = self.token_emb_layer(qidx)

        # (, 1, 300), (, 1, h) => (, 1, h+300)
        qemb = tf.concat([tokenemb, c0], -1)

        # (, 1, h), (, h)
        qdec, s1 = self.gru(qemb, initial_state=s0)

        # (, 1, h) => (, 1, v)
        with tf.device("cpu:0"):
            y = self.dec(qdec)

        state["s0"] = s1
        state["attn_weights"] = attn_weights
        return y, state
