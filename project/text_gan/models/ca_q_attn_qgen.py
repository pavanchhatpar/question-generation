import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np

from ..layers import CA_Q_Encoder, CA_Q_Decoder
from ..data.qgen_ca_q import CA_Qcfg


class CA_Q_AttnQGen:
    def __init__(
            self, word_emb_mat, qword_emb_mat):
        encoder = CA_Q_Encoder(word_emb_mat)
        decoder = CA_Q_Decoder(qword_emb_mat)
        cidx = layers.Input(
            shape=(CA_Qcfg.CSEQ_LEN,), name="Context-Tokens")
        aidx = layers.Input(
            shape=(CA_Qcfg.CSEQ_LEN,), name="Answer-Markers")
        qidx = layers.Input(
            shape=(None,), name="Question-Tokens")
        decoder_state_input = layers.Input(
            shape=(32,), name="Decoder-state")
        encoder_op_input = layers.Input(
            shape=(CA_Qcfg.CSEQ_LEN, 32), name="Encoder-output")

        dec_init_state, enc_output = encoder((cidx, aidx))
        self.encoder = Model(
            [cidx, aidx], [dec_init_state, enc_output],
            name="Attn-Gen-Enc")

        ypred, dec_next_state = decoder(
            (qidx, decoder_state_input, encoder_op_input))
        self.decoder = Model(
            [qidx, decoder_state_input, encoder_op_input],
            [ypred, dec_next_state], name="Attn-Gen-Dec")

        y, _ = decoder(
            (qidx, dec_init_state, enc_output))
        self.model = Model(
            [cidx, aidx, qidx], y, name="Attn-Gen")

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
        target = tf.fill([input[0].shape[0], 1], 2)  # <S> token

        counter = 0
        op = target
        while counter < CA_Qcfg.QSEQ_LEN:
            y, h = self.decoder.predict([target, states_value, enc_op])
            y = np.reshape(y, [input[0].shape[0], CA_Qcfg.QVOCAB_SIZE])
            # sampled_token_index = tf.multinomial(predictions, num_samples=1)
            sampled_token_index = tf.argmax(
                y[0], output_type=tf.int32)
            sampled_token_index = tf.expand_dims([sampled_token_index], 0)
            op = tf.concat([op, sampled_token_index], 1)
            target = sampled_token_index
            states_value = h
            counter += 1
            if target == 3:  # EOS token
                break
        return op

    def predict(self, dataset):
        ret_val = None
        for X, y in dataset:
            op = self._decode_sequence([X[0], X[1]])
            if ret_val is None:
                ret_val = op
                continue
            ret_val = tf.concat([ret_val, op], 0)
        return ret_val

    def plot_model(self, loc):
        return tf.keras.utils.plot_model(
            self.model, loc, show_shapes=True)
