import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from typing import Dict, Any, Tuple
from copynet_tf import Vocab
from copynet_tf.layers import FixedEmbedding

from ..config import cfg
from ..features import NERTagger, PosTagger


class CANP_PreQC_Encoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 ner: NERTagger,
                 pos: PosTagger,
                 context_emb_layer: FixedEmbedding,
                 preq_emb_layer: FixedEmbedding,
                 **kwargs: Dict[str, Any]) -> None:
        super(CANP_PreQC_Encoder, self).__init__(**kwargs)

        self.token_emb_layer = context_emb_layer
        self.preq_emb_layer = preq_emb_layer
        self.ner_emb_layer = layers.Embedding(len(ner.tags2idx), 3)
        self.pos_emb_layer = layers.Embedding(len(pos.tags2idx), 3)
        # uses BIO tagging
        self.ans_emb_layer = layers.Embedding(3, 3)

        # bi-gru
        self.bigru = layers.Bidirectional(layers.GRU(
            cfg.HIDDEN_DIM//2, return_sequences=True, return_state=True))
        self.qbigru = layers.Bidirectional(layers.GRU(
            cfg.HIDDEN_DIM//2))
        self.final = layers.Dense(cfg.HIDDEN_DIM, activation="tanh")

    def call(self,
             cis: tf.Tensor,
             ans: tf.Tensor,
             ner: tf.Tensor,
             pos: tf.Tensor,
             preq: tf.Tensor,
             enc_hidden: tf.Tensor,
             training: bool = False) -> Tuple[
                 tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.device("cpu:0"):
            # shape: (batch_size, cseq_len, 300)
            tokenemb = self.token_emb_layer(cis)
            # shape: (batch_size, cseq_len)
            source_mask = self.token_emb_layer.compute_mask(cis)
            # shape: (batch_size, qseq_len, 300)
            preqemb = self.preq_emb_layer(preq)
            # shape: (batch_size, qseq_len)
            preq_mask = self.preq_emb_layer.compute_mask(preq)

        # shape: (batch_size, cseq_len, 3)
        neremb = self.ner_emb_layer(ner)

        # shape: (batch_size, cseq_len, 3)
        posemb = self.pos_emb_layer(pos)

        # shape: (batch_size, cseq_len, 3)
        ansemb = self.ans_emb_layer(ans)

        # shape: (batch_size, cseq_len, 309)
        gruin = tf.concat([tokenemb, neremb, posemb, ansemb], -1)

        # shape: (batch_size, cseq_len, hidden_dim),
        # (batch_size, hidden_dim//2),
        # (batch_size, hidden_dim//2)
        hd, hD_1, hD_2 = self.bigru(
            gruin, initial_state=enc_hidden, mask=source_mask)

        # shape: (batch_size, hidden_dim)
        preq_final = self.qbigru(preqemb, mask=preq_mask)

        # shape: (batch_size, hidden_dim)
        hD = tf.concat([hD_1, hD_2, preq_final], -1)
        hD = self.final(hD)

        return hd, hD, source_mask, tokenemb

    def initialize_hidden_size(self, batch_sz):
        return [
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2)),
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2))
        ]
