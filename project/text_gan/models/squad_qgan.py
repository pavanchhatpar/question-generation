import tensorflow as tf
import numpy as np
import ujson as json
import os

from ..layers.fixed_embedding import FixedEmbedding


class QGAN:
    def __init__(self, config, lr=1e-3, try_loading=True):
        word_emb_mat = np.array(json.load(open(config.WORDEMBMAT, "r")))
        # Context inputs
        context = tf.keras.layers.Input(
            shape=(config.MAX_CONTEXT_LEN,), name="Context-Tokens")
        discourse_markers = tf.keras.layers.Input(
            shape=(config.MAX_CONTEXT_LEN,), name="Context-Discourse-Markers")
        latent_vector = tf.keras.layers.Input(
            shape=(config.LATENT_DIM,), name="Latent-Vector")

        # Encoder
        context_emb = FixedEmbedding(
            word_emb_mat, config.MAX_CONTEXT_LEN, name="Context-Embedding")(
                context)
        enc_x1 = tf.keras.layers.GRU(
            32, return_sequences=True, name="Context-Encoder-1")(context_emb)
        enc_x1, enc_x1_state = tf.keras.layers.GRU(
            32, return_state=True, name="Context-Encoder-2")(enc_x1)

        enc_x = tf.keras.layers.Multiply()([enc_x1_state, latent_vector])

        self.encoder = tf.keras.Model(
            [context, discourse_markers, latent_vector],
            enc_x, name="QGAN-Enc")

        # Decoder
        question_tokens = tf.keras.layers.Input(
            shape=(config.MAX_QLEN,), name="Question-Tokens")
        decoder_state_input = tf.keras.layers.Input(
            shape=(32,), name="Decoder-state")

        question_emb = tf.keras.layers.Embedding(
            config.QVOCAB_SIZE, 32, name="Question-Embedding")(question_tokens)
        decoder_1 = tf.keras.layers.GRU(
            32, return_sequences=True, name="GRU-Decoder-1")
        decoder_2 = tf.keras.layers.GRU(
            32, return_sequences=True, return_state=True, name="GRU-Decoder-2")
        decoder_dense = tf.keras.layers.Dense(
            config.QVOCAB_SIZE, activation='softmax', name="Dense-Decoder")

        dec_ypred = decoder_1(question_emb, initial_state=decoder_state_input)
        dec_ypred, dec_ypred_state = decoder_2(dec_ypred)
        self.decoder = tf.keras.Model(
            [question_tokens, decoder_state_input],
            [dec_ypred, dec_ypred_state], name="QGAN-Dec")

        # Training model
        dec_y = decoder_1(question_emb, initial_state=enc_x)
        dec_y, _ = decoder_2(dec_y)
        y = decoder_dense(dec_y)
        self.model = tf.keras.Model(
            [context, discourse_markers, latent_vector, question_tokens],
            y, name="QGAN-Trainer")

        if os.path.exists(config.MODELSAVELOC):
            self.model.load_weights(config.MODELSAVELOC)
        else:
            self.model.compile(
                tf.keras.optimizers.Adam(lr),
                'sparse_categorical_crossentropy'
            )

        self.config = config

    def fit(self, dataset, epochs):
        self.model.fit(dataset, epochs=epochs)

    def save(self):
        self.model.save_weights(self.config.MODELSAVELOC)

    def predict(self, dataset):
        # enc_state = self.encoder.predict(dataset)
        # target = np.array([4998])
        pass

    def print_model(self):
        pass
