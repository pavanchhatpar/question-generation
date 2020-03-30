import tensorflow as tf
import numpy as np

from ..layers import Encoder, Decoder


class AttnGen:
    def __init__(
            self, word_emb_mat, qword_emb_mat,
            qword2idx, idx2qword, config):
        self.qword2idx = qword_emb_mat
        self.idx2qword = idx2qword
        self.config = config
        encoder = Encoder(word_emb_mat, config)
        decoder = Decoder(qword_emb_mat, config)
        c_idx = tf.keras.layers.Input(
            shape=(config.MAX_CONTEXT_LEN,), name="Context-Tokens")
        c_dis = tf.keras.layers.Input(
            shape=(config.MAX_CONTEXT_LEN,), name="Context-Discourse-Markers")
        z = tf.keras.layers.Input(
            shape=(config.LATENT_DIM,), name="Latent-Vector")
        q_idx = tf.keras.layers.Input(
            shape=(None,), name="Question-Tokens")
        decoder_state_input = tf.keras.layers.Input(
            shape=(32,), name="Decoder-state")
        encoder_op_input = tf.keras.layers.Input(
            shape=(config.MAX_CONTEXT_LEN, 32), name="Encoder-output")

        dec_init_state, enc_output = encoder((c_idx, c_dis, z))
        self.encoder = tf.keras.Model(
            [c_idx, c_dis, z], [dec_init_state, enc_output],
            name="Attn-Gen-Enc")

        ypred, dec_next_state = decoder(
            (q_idx, decoder_state_input, encoder_op_input))
        self.decoder = tf.keras.Model(
            [q_idx, decoder_state_input, encoder_op_input],
            [ypred, dec_next_state], name="Attn-Gen-Dec")

        y, _ = decoder(
            (q_idx, dec_init_state, enc_output))
        self.model = tf.keras.Model(
            [c_idx, c_dis, z, q_idx], y, name="Attn-Gen")

    def fit(self, dataset, epochs, callbacks):
        return self.model.fit(dataset, epochs=epochs, callbacks=callbacks)

    def save(self, loc):
        self.model.save_weights(loc)

    def load(self, loc):
        self.model.load_weights(loc)

    def _decode_sequence(self, input):
        # Encode the input as state vectors.
        states_value, enc_op = self.encoder.predict(input)

        # Start output sequence
        target = tf.fill([input[0].shape[0], 1], 4998)

        counter = 0
        op = target
        while counter < self.config.MAX_QLEN:
            y, h = self.decoder.predict([target, states_value, enc_op])
            y = np.reshape(y, [input[0].shape[0], self.config.QVOCAB_SIZE])
            sampled_token_index = tf.argmax(
                y, output_type=tf.int32, axis=1)
            sampled_token_index = [sampled_token_index]
            # sampled_word = self.idx2qword[sampled_token_index]
            # if op is None:
            #     op =
            op = tf.concat([op, sampled_token_index], 1)
            target = sampled_token_index
            states_value = h
            counter += 1
        return op

    def predict(self, dataset):
        ret_val = None
        for X, y in dataset:
            op = self._decode_sequence([X[0], X[1], X[2]])
            if ret_val is None:
                ret_val = op
                continue
            ret_val = tf.concat([ret_val, op], 0)
        return ret_val

    def plot_model(self, loc):
        return tf.keras.utils.plot_model(
            self.model, loc, show_shapes=True)


# class AttnGen(tf.keras.Model):
#     def __init__(
#             self, word_emb_mat, qword_emb_mat,
#             qword2idx, idx2qword, config, **kwargs):
#         super(AttnGen, self).__init__(**kwargs)
#         self.qword2idx = qword_emb_mat
#         self.idx2qword = idx2qword
#         self.encoder = Encoder(word_emb_mat, config)
#         self.decoder = Decoder(qword_emb_mat, config)
#         self.config = config

#     def call(self, inputs, training=None):
#         c_idx, c_dis, z, q_idx = inputs
#         s0, hd = self.encoder((c_idx, c_dis, z))
#         y, _ = self.decoder((q_idx, s0, hd))
#         return y
#         # else:
#         #     q_idx = tf.fill([c_idx.shape[0], 1], 4998)
#         #     # done = False
#         #     # while not q_idx == self.qword2idx['<END>']\
#         #     #         and not len(op) == self.config.MAX_QLEN:
#         #     y, st = self.decoder((q_idx, s0, hd))
#         #     q_idx = tf.random.categorical(
#                           y[0], num_samples=1, dtype=tf.int32)
#         #     # q_idx = tf.convert_to_tensor([q_idx])
#         #     # done = q_idx == self.qword2idx['<END>']\
#         #     #     or len(op) == self.config.MAX_QLEN
#         #     return q_idx, st
