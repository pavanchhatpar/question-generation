import tensorflow as tf

from .fixed_embedding import FixedEmbedding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, word_emb_mat, config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.config = config
        self.word_emb_mat = word_emb_mat

    def build(self, input_shape):
        self.context_embedding = FixedEmbedding(
            self.word_emb_mat,
            self.config.MAX_CONTEXT_LEN, name="Context-Embedding")
        self.bigru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            16, return_sequences=True, name="Context-Encoder-1"))
        self.bigru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                16, return_sequences=True,
                return_state=True, name="Context-Encoder-2"))
        # self.qbigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        #         16, return_state=True, name="Context-Encoder-2"))
        self.dense_enc = tf.keras.layers.Dense(32, activation='tanh')

    def call(self, inputs):
        c_idx, c_dis, z = inputs
        c_embs = self.context_embedding(c_idx)

        c_int_enc = self.bigru_1(c_embs)
        c_att, c_final_fenc, c_final_benc = self.bigru_2(c_int_enc)
        c_final_enc = tf.concat([c_final_fenc, c_final_benc], -1)

        c_dis = tf.cast(tf.expand_dims(c_dis, -1), tf.float32)
        dis_enc = tf.reduce_mean(tf.multiply(c_att, c_dis), 1)

        # q_embs = self.question_embedding(q_idx)
        # q_enc = self.qbigru(q_embs)
        op = tf.concat([c_final_enc, dis_enc, z], -1)
        op = self.dense_enc(op)
        return op, c_att
