import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from typing import Dict, Any
from copynet_tf import Vocab

from ..config import cfg
from ..features import NERTagger, PosTagger


class CANPZ_Q_Encoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 ner: NERTagger,
                 pos: PosTagger,
                 context_emb_layer,
                 **kwargs: Dict[str, Any]):
        super(CANPZ_Q_Encoder, self).__init__(**kwargs)

        # embedding layers
        # with tf.device("cpu:0"):
        self.token_emb_layer = context_emb_layer
        self.ner_emb_layer = layers.Embedding(len(ner.tags2idx), 4)
        self.pos_emb_layer = layers.Embedding(len(pos.tags2idx), 5)

        self.input_projection_layer = layers.Dense(300)

        # bi-gru
        self.bigru = layers.Bidirectional(layers.GRU(
            cfg.HIDDEN_DIM//2, return_sequences=True, return_state=True))

        # dense layers
        self.latent_reparam = layers.Dense(cfg.LATENT_DIM*2, activation="relu")
        self.enc = layers.Dense(cfg.HIDDEN_DIM, activation="tanh")

    def call(self, cidx, aidx, ner, pos, enc_hidden, training=None):
        with tf.device("cpu:0"):
            # (, 250) => (, 250, 300)
            tokenemb = self.token_emb_layer(cidx)

            # shape: (batch_size, source_seq_len)
            source_mask = self.token_emb_layer.compute_mask(cidx)

        # (, 250) => (, 250, 4)
        neremb = self.ner_emb_layer(ner)

        # (, 250) => (, 250, 5)
        posemb = self.pos_emb_layer(pos)

        # (, 250, 300), (, 250, 4), (, 250, 5) => (, 250, 309)
        cemb = tf.concat([tokenemb, neremb, posemb], -1)

        # shape: (batch_size, source_seq_len, 300)
        cemb = self.input_projection_layer(cemb)

        # (, 250, 300) => (, 250, h), (, h//2), (, h//2)
        hd, cfinalfenc, cfinalbenc = self.bigru(
            cemb, initial_state=enc_hidden, mask=source_mask)

        # (, h//2), (, h//2) => (, h)
        hD = tf.concat([cfinalfenc, cfinalbenc], -1)

        # (, 250) => (, 250, 1)
        aidx = tf.cast(tf.expand_dims(aidx, -1), tf.float32)

        # (, h)
        ha = tf.reduce_sum(tf.multiply(hd, aidx), 1)
        alen = tf.maximum(tf.reduce_sum(aidx, 1), 1)
        ha = ha / alen

        # (, h*2)
        meanlogvar = tf.concat([hD, ha], -1)

        # (, h*2)
        meanlogvar = self.latent_reparam(meanlogvar)

        if training:
            meanlogvar = tf.nn.dropout(meanlogvar, rate=cfg.DROPOUT)

        # (, h*2) => (, l), (, l)
        mean, logvar = tf.split(
            meanlogvar, num_or_size_splits=2, axis=1)

        # reparameterization
        # (, l)
        eps = tf.random.normal(shape=mean.shape)
        # (, l)
        z = eps * tf.exp(logvar * 0.5) + mean

        # (, h), (, h), (, l) => (, h)
        s0 = self.enc(tf.concat([hD, ha, z], -1))

        if training:
            s0 = tf.nn.dropout(s0, rate=cfg.DROPOUT)

        return s0, hd, mean, logvar, z, source_mask

    def initialize_hidden_size(self, batch_sz):
        return [
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2)),
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2))
        ]
