import tensorflow as tf
import numpy as np
import ujson as json

from ..layers.fixed_embedding import FixedEmbedding


class QGAN:
    def __init__(self, config, lr=1e-3, try_loading=True):
        word_emb_mat = np.array(json.load(open(config.WORDEMBMAT, "r")))
        self.qword2idx = json.load(open(config.QWORD2IDX, "r"))
        self.idx2qword = np.full(config.QVOCAB_SIZE, "<UNK>", dtype='object')
        for word, idx in self.qword2idx.items():
            self.idx2qword[idx] = word
        self.word2idx = json.load(open(config.WORD2IDX, "r"))
        self.idx2word = np.full(len(self.word2idx), "<UNK>", dtype='object')
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
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
        enc_x1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            16, return_sequences=True, name="Context-Encoder-1"))(context_emb)
        enc_x1, enc_x1_fstate, enc_x1_bstate = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                16, return_state=True, name="Context-Encoder-2"))(enc_x1)
        enc_x1_state = tf.keras.layers.Concatenate(name="State-Concatenate")(
            [enc_x1_fstate, enc_x1_bstate])
        enc_x = tf.keras.layers.Add(name="Add-random")(
            [enc_x1_state, latent_vector])

        self.encoder = tf.keras.Model(
            [context, discourse_markers, latent_vector],
            enc_x, name="QGAN-Enc")

        # Decoder
        question_tokens = tf.keras.layers.Input(
            shape=(None,), name="Question-Tokens")
        decoder_state_input = tf.keras.layers.Input(
            shape=(32,), name="Decoder-state")

        question_emb = tf.keras.layers.Embedding(
            config.QVOCAB_SIZE, 32, name="Question-Embedding")(question_tokens)
        decoder_1 = tf.keras.layers.GRU(
            32, return_sequences=True, return_state=True, name="GRU-Decoder-1")
        decoder_dense = tf.keras.layers.Dense(
            config.QVOCAB_SIZE, activation='softmax', name="Dense-Decoder")

        dec_ypred, dec_ypred_state = decoder_1(
            question_emb, initial_state=decoder_state_input)
        ypred = decoder_dense(dec_ypred)
        self.decoder = tf.keras.Model(
            [question_tokens, decoder_state_input],
            [ypred, dec_ypred_state], name="QGAN-Dec")

        # Training model
        dec_y, _ = decoder_1(question_emb, initial_state=enc_x)
        y = decoder_dense(dec_y)
        self.model = tf.keras.Model(
            [context, discourse_markers, latent_vector, question_tokens],
            y, name="QGAN-Trainer")
        self.config = config
        self.model.compile(
            tf.keras.optimizers.Adam(lr),
            'sparse_categorical_crossentropy'
        )
        if try_loading:
            self.model.load_weights(config.MODELSAVELOC)

    def fit(self, dataset, epochs, **kwargs):
        return self.model.fit(dataset, epochs=epochs, **kwargs)

    def save(self):
        self.model.save_weights(self.config.MODELSAVELOC)

    def _decode_sequence(self, input):
        # Encode the input as state vectors.
        states_value = self.encoder.predict(input)

        # Start output sequence
        target = np.array([[4998]])

        decoded_sentence = ''
        counter = 0
        done = False

        while not done:
            output_tokens, h = self.decoder.predict([target, states_value])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.idx2qword[sampled_token_index]
            decoded_sentence += sampled_word + " "

            target = np.array([sampled_token_index])
            states_value = h
            counter += 1
            done = sampled_word == "<END>"\
                or counter >= self.config.MAX_QLEN
        return decoded_sentence

    def predict(self, dataset):
        ret_val = []
        for X, y in dataset:
            ret_val.append(self._decode_sequence([X[0], X[1], X[2]]))
        return ret_val

    def plot_model(self):
        return tf.keras.utils.plot_model(
            self.model, "/tf/data/model.png", show_shapes=True)
