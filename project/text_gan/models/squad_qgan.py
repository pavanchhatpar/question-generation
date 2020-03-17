import tensorflow as tf
import numpy as np
import ujson as json
from ..layers.fixed_embedding import FixedEmbedding

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_func(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_model(config):
    word_emb_mat = np.array(json.load(open(config.WORDEMBMAT, "r")))
    rand_in = tf.keras.layers.Input(
        shape=(config.LATENT_DIM,), name="Latent-input")
    context_idx = tf.keras.layers.Input(shape=(256,), name="Context-IDs")
    context_dis = tf.keras.layers.Input(shape=(256,), name="Discourse-markers")
    context_embs = FixedEmbedding(
        word_emb_mat, 256, name="Glove-embeddings")(context_idx)
    context_embs = tf.keras.layers.GRU(
        32, return_sequences=True, name="GRU-encoder-1")(context_embs)
    context_embs = tf.keras.layers.GRU(32, name="GRU-encoder-2")(context_embs)
    context_embs = tf.keras.layers.RepeatVector(
        config.MAX_QLEN, name="Repeater")(context_embs)
    context_dis_enc = tf.keras.layers.Dense(
        32, activation='tanh', name="Discourse-encoder-1")(context_dis)
    context_dis_enc = tf.keras.layers.Dense(
        32, activation='relu', name="Discourse-encoder-2")(context_dis_enc)
    enc = tf.keras.layers.Add(name="Encoder-mixer")(
        [context_dis_enc, context_embs, rand_in])
    dec = tf.keras.layers.Dense(config.QVOCAB_SIZE, name="Question-Decoder")(
        enc)
    model = tf.keras.Model(
        inputs=[context_idx, context_dis, rand_in],
        outputs=dec, name="SQuAD-QGAN")
    model.compile(
        tf.keras.optimizers.Adam(),
        loss_object
    )
    return model
