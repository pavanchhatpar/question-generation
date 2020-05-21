import tensorflow as tf
import numpy as np
from absl import logging
from typing import Dict, Any
from copynet_tf.layers import FixedEmbedding, FixedDense
# from copynet_tf.metrics import compute_bleu
from copynet_tf.search import BeamSearch
from copynet_tf import CopyNetDecoder, Vocab
from copynet_tf.util import prep_y_true
from tensorflow.keras import Model

from ..layers import CANP_QC_Encoder
from ..features import NERTagger, PosTagger
from ..config import cfg


class CANP_QC(Model):
    def __init__(self,
                 vocab: Vocab,
                 ner: NERTagger,
                 pos: PosTagger,
                 **kwargs: Dict[str, Any]) -> None:
        super(CANP_QC, self).__init__(**kwargs)
        self.vocab = vocab
        copy_token = "@COPY@"
        self.vocab.add_token(copy_token, "target")
        # with tf.device("cpu:0"):
        context_vocab = tf.convert_to_tensor(
            vocab.get_embedding_matrix("source"))
        question_vocab = tf.convert_to_tensor(
            vocab.get_embedding_matrix("target"))
        context_emb_layer = FixedEmbedding(
            context_vocab,
            cfg.CSEQ_LEN, mask_zero=True)
        question_dec_layer = FixedDense(
            question_vocab.shape[0],
            weights=[
                tf.transpose(question_vocab),
                np.zeros(question_vocab.shape[0])])
        self.encoder = CANP_QC_Encoder(vocab, ner, pos, context_emb_layer)
        self.searcher = BeamSearch(
            cfg.BEAM_WIDTH,
            self.vocab.get_token_id(self.vocab._end_token, "target"),
            cfg.QPRED_LEN)
        self.decoder = CopyNetDecoder(
            vocab, cfg.HIDDEN_DIM, self.searcher, question_dec_layer,
            copy_token=copy_token)
        self._train_counter = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="train_counter")

    @tf.function
    def call(self, X, y=(None, None), training=False):
        cis, cit, ans, ner, pos = X
        qit, qis = y
        enc_hidden = self.encoder.initialize_hidden_size(cis.shape[0])
        hd, hD, mask = self.encoder(
            cis, ans, ner, pos, enc_hidden, training)
        state = {
            "encoder_output": hd,
            "encoder_final_output": hD
        }
        output_dict = self.decoder(
            cis, cit, mask, state, qit, qis, training=training)
        return output_dict

    def debug(self, cis, qit, ans, attn_weights, selective_weights):
        logging.debug("**** Train counter: %d ****", self._train_counter)
        k = 5
        # shape (samples, max_decoding_steps, 5)
        max_attn, indices = tf.math.top_k(attn_weights, k=k)
        # shape (samples, max_decoding_steps, 5)
        cis_tokens = tf.gather(cis, indices, axis=1, batch_dims=1)

        # shape (samples, max_decoding_steps, 5)
        max_attn1, indices1 = tf.math.top_k(selective_weights, k=k)
        # shape (samples, max_decoding_steps, 5)
        cis_tokens1 = tf.gather(cis, indices1, axis=1, batch_dims=1)
        qt = self.vocab.inverse_transform(qit.numpy(), "target")
        for i in range(cis.shape[0]):
            a = ''
            for j, aj in enumerate(ans[i]):
                if aj == 0:
                    continue
                a += self.vocab.get_token_text(
                    cis[i][j].numpy(), "source") + ' '
            logging.debug(f"answer:\n{a}")
            ct = self.vocab.inverse_transform(cis_tokens[i].numpy(), "source")
            logging.debug(f"context tokens:\n{ct}")
            logging.debug(f"attention values:\n{max_attn[i]}")
            logging.debug("")

            ct = self.vocab.inverse_transform(cis_tokens1[i].numpy(), "source")
            logging.debug(f"context tokens:\n{ct}")
            logging.debug(f"selection values:\n{max_attn1[i]}")
            logging.debug("")

            logging.debug(f"Target words:\n{qt[i]}")
            logging.debug("\n")

    @tf.function
    def train_step(self, data):
        X, y = data
        training = tf.constant(True)
        cis, cit, ans, ner, pos = X
        qit, qis = y

        target_vocab_size = self.vocab.get_vocab_size("target")
        unk_index = self.vocab.get_token_id(self.vocab._unk_token, "source")
        start_index = self.vocab.get_token_id(
            self.vocab._start_token, "source")
        end_index = self.vocab.get_token_id(self.vocab._end_token, "source")

        y_true = prep_y_true(
            cis, qit, qis, target_vocab_size,
            unk_index, start_index, end_index)
        with tf.GradientTape() as tape:
            output_dict = self(X, y, training)
            # shape: ()
            loss = self.compiled_loss(
                y_true,
                output_dict['ypred'])
            if (self._train_counter % 10 == 0
                and (logging.get_absl_logger().getEffectiveLevel()
                     == logging.converter.STANDARD_DEBUG)):
                samples = 3
                tf.py_function(
                    self.debug,
                    [
                        cis[:samples],
                        qit[:samples],
                        ans[:samples],
                        output_dict["attentive_weights"][:samples],
                        output_dict["selective_weights"][:samples]
                    ],
                    [], name="Debug"
                )
            gradients = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(
                gradients, self.trainable_variables))

        self.compiled_metrics.update_state(
                y_true,
                output_dict['ypred'])
        self._train_counter.assign_add(1)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        X, y = data
        training = tf.constant(True)
        cis, cit, ans, ner, pos = X
        qit, qis = y

        target_vocab_size = self.vocab.get_vocab_size("target")
        unk_index = self.vocab.get_token_id(self.vocab._unk_token, "source")
        start_index = self.vocab.get_token_id(
            self.vocab._start_token, "source")
        end_index = self.vocab.get_token_id(self.vocab._end_token, "source")

        y_true = prep_y_true(
            cis, qit, qis, target_vocab_size,
            unk_index, start_index, end_index)

        output_dict = self(X, y, training)
        self.compiled_loss(
            y_true,
            output_dict['ypred'])

        self.compiled_metrics.update_state(
                y_true,
                output_dict['ypred'])
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data):
        X, _ = data
        return self(X)
