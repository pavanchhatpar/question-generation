import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Any
from copynet_tf.layers import FixedEmbedding, FixedDense
# from copynet_tf.metrics import compute_bleu
from copynet_tf.search import BeamSearch
from copynet_tf import CopyNetDecoder, Vocab

from .canp_qc_encoder import CANP_QC_Encoder
from ..features import NERTagger, PosTagger
from ..config import cfg


class CANP_QC:
    def __init__(self,
                 vocab: Vocab,
                 ner: NERTagger,
                 pos: PosTagger,
                 **kwargs: Dict[str, Any]) -> None:
        self.vocab = vocab
        with tf.device("cpu:0"):
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
            vocab, cfg.HIDDEN_DIM, self.searcher, question_dec_layer)
        self.logger = logging.getLogger(__name__)

    @tf.function
    def forward_step(self, cis, cit, ans, ner, pos, qit, qis,
                     enc_hidden, epoch_no, batch_no, training):
        hd, hD, mask, cemb = self.encoder(
            cis, ans, ner, pos, enc_hidden, training)
        state = {
            "encoder_output": hd,
            "encoder_final_output": hD
        }
        output_dict = {}
        if training:
            output_dict = self.decoder(
                cis, cit, cemb, mask, state, qit, qis, training=training)
        else:
            output_dict = self.decoder.predict(
                cis, cit, cemb, mask, state, qit, qis, training=training)
        return output_dict

    def debug(
            self, cis, qit, ans, attn_weights,
            selective_weights, epoch, batch):
        k = 5
        self.logger.debug(f"**** Epoch {epoch} Batch {batch} ****")
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
            self.logger.debug(f"answer:\n{a}")
            ct = self.vocab.inverse_transform(cis_tokens[i].numpy(), "source")
            self.logger.debug(f"context tokens:\n{ct}")
            self.logger.debug(f"attention values:\n{max_attn[i]}")
            self.logger.debug("")

            ct = self.vocab.inverse_transform(cis_tokens1[i].numpy(), "source")
            self.logger.debug(f"context tokens:\n{ct}")
            self.logger.debug(f"selection values:\n{max_attn1[i]}")
            self.logger.debug("")

            self.logger.debug(f"Target words:\n{qt[i]}")
            self.logger.debug("\n")

    @tf.function
    def train_step(self, X, y, epoch_no, batch_no):
        training = tf.constant(True)
        with tf.GradientTape() as tape:
            cis, cit, ans, ner, pos = X
            qit, qis = y
            enc_hidden = self.encoder.initialize_hidden_size(
                qit.shape[0])
            output_dict = self.forward_step(
                cis, cit, ans, ner, pos, qit, qis,
                enc_hidden, epoch_no, batch_no, training)
            loss = output_dict["loss"]
            if (
                batch_no % 10 == 0
                    and self.logger.getEffectiveLevel() == logging.DEBUG):
                samples = 3
                tf.py_function(
                    self.debug,
                    [
                        cis[:samples],
                        qit[:samples],
                        ans[:samples],
                        output_dict["attentive_weights"][:samples],
                        output_dict["selective_weights"][:samples],
                        epoch_no,
                        batch_no
                    ],
                    [], name="Debug"
                )
            vars = (self.encoder.trainable_variables
                    + self.decoder.trainable_variables)

            gradients = tape.gradient(loss, vars)

            self.optimizer.apply_gradients(zip(gradients, vars))

        return loss

    def fit(
            self, dataset, epochs,
            save_loc, eval_set=None, warm_start=False):
        if not warm_start:
            self.optimizer = tf.keras.optimizers.Adam(
                cfg.LR, clipnorm=cfg.CLIP_NORM)
            ckpt_saver = tf.train.Checkpoint(
                optimizer=self.optimizer,
                encoder=self.encoder,
                decoder=self.decoder)
            ckpt_manager = tf.train.CheckpointManager(
                ckpt_saver, save_loc, max_to_keep=cfg.CKPT_COUNT)
        else:
            self.optimizer, ckpt_saver, ckpt_manager = self._load(save_loc)
        training = tf.constant(True)
        for epoch in tf.range(epochs):
            eloss = tf.constant(0, dtype=tf.float32)
            i = tf.constant(0, dtype=tf.float32)
            with tqdm(
                    dataset, desc=f"Epoch {epoch.numpy()+1}/{epochs}") as pbar:
                for X, y in pbar.iterable:
                    bloss = self.train_step(X, y, epoch+1, i+1)
                    pbar.update(1)
                    i += 1
                    eloss = (eloss*(i-1) + bloss)/i
                    metrics = {"train-loss": f"{eloss:.4f}"}
                    pbar.set_postfix(metrics)
                # calculate train metrics on one batch
                bleus = self.evaluate(dataset.take(1))
                metrics["train-bleu"] = bleus["bleu"]
                metrics["train-bleu-smooth"] = bleus["bleu-smooth"]
                pbar.set_postfix(metrics)
                if eval_set is not None:
                    vloss = tf.constant(0, dtype=tf.float32)
                    n = tf.constant(0, dtype=tf.float32)
                    for X, y in eval_set:
                        cis, cit, ans, ner, pos = X
                        qit, qis = y
                        enc_hidden = self.encoder.initialize_hidden_size(
                            qit.shape[0])
                        out = self.forward_step(
                            cis, cit, ans, ner, pos, qit, qis,
                            enc_hidden, epoch+1, n+1, training)
                        vloss += out["loss"]
                        n += 1
                    metrics['val-loss'] = f"{vloss/n:.4f}"
                    bleus = self.evaluate(eval_set)
                    metrics["val-bleu"] = bleus["bleu"]
                    metrics["val-bleu-smooth"] = bleus["bleu-smooth"]
                    pbar.set_postfix(metrics)
            ckpt_manager.save()

    def evaluate(self, dataset):
        nottraining = tf.constant(False)
        epoch = tf.constant(1, dtype=tf.int32)
        n = tf.constant(0, dtype=tf.float32)
        # shape: (batch_size, max_seq_len)
        preds = None
        # shape: (batch_size, 1, max_seq_len)
        references = None
        ignore_tokens = [2, 3]
        ignore_all_tokens_after = 3
        target_vocab_size = self.vocab.get_vocab_size("target")
        for X, y in dataset:
            cis, cit, ans, ner, pos = X
            qit, qis = y
            enc_hidden = self.encoder.initialize_hidden_size(
                qit.shape[0])
            out = self.forward_step(
                cis, cit, ans, ner, pos, qit, qis,
                enc_hidden, epoch, n+1, nottraining)
            n += 1
            predictions = out["predictions"]
            if preds is None:
                preds = predictions[:, 0]
                preds_sub = preds - target_vocab_size
                preds_sub = tf.where(
                    preds_sub < 0, 0, preds_sub)
                preds_sub = tf.gather(
                    cis, preds_sub, axis=-1, batch_dims=1)
                preds = tf.where(
                    preds > target_vocab_size, preds_sub, preds)

                references = tf.where(qit == 1, qis, qit)
                references = tf.expand_dims(references, 1)
            else:
                refs = tf.where(qit == 1, qis, qit)
                refs = tf.expand_dims(refs, 1)
                references = tf.concat(
                    [references, refs], axis=0)

                pred = predictions[:, 0]
                pred_sub = pred - target_vocab_size
                pred_sub = tf.where(
                    pred_sub < 0, 0, pred_sub)
                pred_sub = tf.gather(
                    cis, pred_sub, axis=-1, batch_dims=1)
                pred = tf.where(
                    pred > target_vocab_size, pred_sub, pred)
                preds = tf.concat(
                    [preds, pred], axis=0)

        bleu = compute_bleu(
            references.numpy(), preds.numpy(),
            ignore_tokens=ignore_tokens,
            ignore_all_tokens_after=ignore_all_tokens_after)[0]
        bleu_smooth = compute_bleu(
            references.numpy(), preds.numpy(), smooth=True,
            ignore_tokens=ignore_tokens,
            ignore_all_tokens_after=ignore_all_tokens_after)[0]

        return {
            "bleu": bleu,
            "bleu-smooth": bleu_smooth
        }

    def predict(self, dataset):
        ret_val = None
        logprobas = None
        nottraining = tf.constant(False)
        training = tf.constant(True)
        epoch = tf.constant(1, dtype=tf.int32)
        n = tf.constant(0, dtype=tf.float32)
        for X, y in dataset:
            cis, cit, ans, ner, pos = X
            qit, qis = y
            enc_hidden = self.encoder.initialize_hidden_size(
                qit.shape[0])
            out = self.forward_step(
                cis, cit, ans, ner, pos, qit, qis,
                enc_hidden, epoch, n+1, nottraining)
            op = out["predictions"]
            logproba = out["predicted_probas"]
            n += 1
            if ret_val is None:
                ret_val = op
                logprobas = logproba
            else:
                ret_val = tf.concat([ret_val, op], 0)
                logprobas = tf.concat([logprobas, logproba], 0)

            # output_dict = self.forward_step(
            #     cis, cit, ans, ner, pos, qit, qis,
            #     enc_hidden, epoch, n, training)
            # loss = output_dict["loss"]
            # samples = cis.shape[0]
            # tf.py_function(
            #     self.debug,
            #     [
            #         cis[:samples],
            #         qit[:samples],
            #         ans[:samples],
            #         output_dict["attentive_weights"][:samples],
            #         output_dict["selective_weights"][:samples],
            #         epoch,
            #         n
            #     ],
            #     [], name="Debug"
            # )

        return ret_val, logprobas

    def _load(self, save_loc):
        optimizer = tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM)
        ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, save_loc, max_to_keep=cfg.CKPT_COUNT)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        return optimizer, ckpt, ckpt_manager

    def load(self, save_loc):
        self._load(save_loc)
