import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
from copynet_tf.layers import FixedEmbedding, FixedDense
from copynet_tf.metrics import compute_bleu
from copynet_tf.search import BeamSearch

from .canpz_q_encoder import CANPZ_Q_Encoder
from .canpz_q_decoder import CANPZ_Q_Decoder
from ..config import cfg


class CANPZ_Q:
    def __init__(self, vocab, ner, pos, **kwargs):
        super(CANPZ_Q, self).__init__(**kwargs)
        with tf.device("cpu:0"):
            context_vocab = tf.convert_to_tensor(
                vocab.get_embedding_matrix("source"))
            question_vocab = tf.convert_to_tensor(
                vocab.get_embedding_matrix("target"))
            context_emb_layer = FixedEmbedding(
                context_vocab,
                cfg.CSEQ_LEN, mask_zero=True)
            question_emb_layer = FixedEmbedding(
                question_vocab, cfg.QSEQ_LEN, mask_zero=True)
            question_dec_layer = FixedDense(
                question_vocab.shape[0],
                weights=[
                    tf.transpose(question_vocab),
                    np.zeros(question_vocab.shape[0])])
        self.encoder = CANPZ_Q_Encoder(vocab, ner, pos, context_emb_layer)
        self.decoder = CANPZ_Q_Decoder(
            vocab, question_emb_layer, question_dec_layer)
        self.logger = logging.getLogger(__name__)
        self.vocab = vocab
        self.searcher = BeamSearch(
            2, self.vocab.get_token_id(self.vocab._end_token, "target"),
            cfg.QSEQ_LEN - 1)

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @staticmethod
    @tf.function
    def lossfn(y_true, y_pred, mean, logvar, z):
        y_true = tf.expand_dims(y_true, 1)  # (batch, 1)
        y_pred = tf.squeeze(y_pred, axis=1)  # (batch, target_vocab_size)
        # shape: (batch, target_vocab_size)
        log_probs = tf.nn.log_softmax(y_pred, axis=1)

        # shape: (batch, 1)
        cross_ent = -tf.gather(log_probs, y_true, axis=1, batch_dims=1)

        # cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=y_pred, labels=y_true)  # (batch, 1)

        # mask PAD token
        mask = tf.math.logical_not(tf.math.equal(y_true, cfg.PAD_ID))
        mask = tf.cast(mask, dtype=cross_ent.dtype)
        cross_ent *= mask  # (batch, 1)
        mask = tf.squeeze(mask, axis=1)

        logpy_z = -tf.reduce_sum(cross_ent, axis=1)
        zero = tf.constant(0, dtype=tf.float32)
        logpz = CANPZ_Q.log_normal_pdf(z, zero, zero)*mask
        logqz_y = CANPZ_Q.log_normal_pdf(z, mean, logvar)*mask

        return -tf.reduce_mean(logpy_z + logpz - logqz_y)

    def debug(
            self, attn_weights, cidx, qidx,
            y, ypred, yi, epoch, batch, step_loss):
        k = 5
        # shape: (samples, source_seq_len)
        sq_attn_weights = tf.squeeze(attn_weights, axis=1)
        # shape: (samples, k) both
        max_attn, indices = tf.math.top_k(
            sq_attn_weights, k=k)
        # shape: (samples, k)
        cidx_tokens = tf.gather(
            cidx, indices, axis=1, batch_dims=1)
        cidx_tokens = self.vocab.inverse_transform(
            cidx_tokens.numpy(), "source")
        self.logger.debug("")
        self.logger.debug(
            f"******************* Epoch {epoch}, Batch {batch}, yi {yi} "
            f"*******************")
        self.logger.debug(
            f"Attn for top {k} tokens in {cidx.shape[0]} samples")
        self.logger.debug(f"context token:\n{cidx_tokens}")
        self.logger.debug(f"context posn:\n{indices}")
        self.logger.debug(f"attn values:\n{max_attn}")
        last_ques_tokens = self.vocab.inverse_transform(
            qidx[:, :1].numpy(), "target")
        target_ques_tokens = self.vocab.inverse_transform(
            y[:, yi].numpy()[:, np.newaxis], "target")
        # shape: (samples, 5)
        maxypred, amaxypred = tf.math.top_k(ypred[:, 0], k=5)
        pred_ques_tokens = self.vocab.inverse_transform(
            amaxypred.numpy(), "target")
        self.logger.debug(
            f"used:\n{last_ques_tokens}\nto predict:\n{target_ques_tokens}")
        self.logger.debug(
            f"predicted:\n{pred_ques_tokens}\n"
            f"with logits:\n{maxypred}")
        self.logger.debug(
            f"Full batch loss for this step: {step_loss}")

    @tf.function
    def forward_step(
            self, X, y, enc_hidden, epoch_no, batch_no, training=None):
        cidx, aidx, ner, pos = X
        loss = 0

        s0, hd, mean, logvar, z, source_mask = self.encoder(
            cidx, aidx, ner, pos, enc_hidden, training)

        state = {
            "s0": s0,
            "hd": hd,
            "source_mask": source_mask
        }

        START_TOKEN_ID = cfg.START_ID
        BATCH_SIZE = cidx.shape[0]
        qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)

        for yi in range(1, y.shape[1]):
            ypred, state = self.decoder(
                qidx, state, training)
            s0 = state["s0"]
            attn_weights = state["attn_weights"]
            step_loss = CANPZ_Q.lossfn(y[:, yi], ypred, mean, logvar, z)
            if (
                self.logger.getEffectiveLevel() == logging.DEBUG
                    and batch_no % 10 == 0
                    and training):
                samples = 3
                tf.py_function(
                    self.debug,
                    [
                        attn_weights[:samples],
                        cidx[:samples],
                        qidx[:samples],
                        y[:samples],
                        ypred[:samples],
                        yi,
                        epoch_no,
                        batch_no,
                        step_loss
                    ], [], name='Debug')

            loss += step_loss

            qidx = tf.expand_dims(y[:, yi], 1)

        loss /= (y.shape[1] - 1)

        return loss

    @tf.function
    def _decode_sequence(self, cidx, aidx, ner, pos, training=False):
        # Encode the input as state vectors.
        enc_hidden = self.encoder.initialize_hidden_size(cidx.shape[0])
        s0, hd, mean, logvar, z, source_mask = self.encoder(
            cidx, aidx, ner, pos, enc_hidden)

        state = {
            "s0": s0,
            "hd": hd,
            "source_mask": source_mask
        }

        # Start output sequence
        START_TOKEN_ID = cfg.START_ID
        BATCH_SIZE = cidx.shape[0]
        qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)  # (batch, 1)
        all_top_k_predictions, log_probabilities = self.searcher.search(
            qidx, state, self.decoder)

        return all_top_k_predictions, log_probabilities

        # counter = 0
        # op = qidx
        # attn = tf.zeros((BATCH_SIZE, 1, cfg.CSEQ_LEN))
        # while counter < cfg.QSEQ_LEN:
        #     y, st, attn_weights = self.decoder(qidx, s0, hd, source_mask)
        #     # sampled_token_index = tf.multinomial(predictions, num_samples=1)
        #     sampled_token_index = tf.argmax(
        #         y, output_type=tf.int32, axis=2)  # (batch, 1)
        #     op = tf.concat([op, sampled_token_index], -1)
        #     attn = tf.concat([attn, attn_weights], -2)
        #     qidx = sampled_token_index
        #     s0 = st
        #     counter += 1
        #     # if qidx == cfg.END_ID:  # EOS token
        #     #     break
        # return op, attn  # (batch, 20), (batch, 20, 250)

    @tf.function
    def train_step(self, X, y, enc_hidden, epoch_no, batch_no):
        with tf.GradientTape() as tape:
            loss = self.forward_step(
                X, y, enc_hidden, epoch_no, batch_no, True)

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
                ckpt_saver, save_loc, max_to_keep=10)
        else:
            self.optimizer, ckpt_saver, ckpt_manager = self._load(save_loc)
        nottraining = tf.constant(False)
        training = tf.constant(True)
        ignore_tokens = [2, 3]
        ignore_all_tokens_after = 3
        for epoch in tf.range(epochs):
            eloss = tf.constant(0, dtype=tf.float32)
            i = tf.constant(0, dtype=tf.float32)
            # shape: (batch_size, max_seq_len)
            preds = None
            # shape: (batch_size, 1, max_seq_len)
            references = None
            with tqdm(
                    dataset, desc=f"Epoch {epoch.numpy()+1}/{epochs}") as pbar:
                for X, y in pbar.iterable:
                    enc_hidden = self.encoder.initialize_hidden_size(
                        y.shape[0])
                    bloss = self.train_step(X, y, enc_hidden, epoch+1, i+1)
                    predictions, logprobas = self._decode_sequence(
                        X[0], X[1], X[2], X[3], training)
                    if preds is None:
                        preds = predictions[:, 0]
                        references = y
                        references = tf.expand_dims(references, 1)
                    else:
                        refs = y
                        refs = tf.expand_dims(refs, 1)
                        references = tf.concat(
                            [references, refs], axis=0)

                        pred = predictions[:, 0]
                        preds = tf.concat(
                            [preds, pred], axis=0)
                    pbar.update(1)
                    i += 1
                    eloss = (eloss*(i-1) + bloss)/i
                    metrics = {"train-loss": f"{eloss:.4f}"}
                    pbar.set_postfix(metrics)
                metrics["train-bleu"] = compute_bleu(
                    references.numpy(), preds.numpy(),
                    ignore_tokens=ignore_tokens,
                    ignore_all_tokens_after=ignore_all_tokens_after)[0]
                metrics["train-bleu-smooth"] = compute_bleu(
                    references.numpy(), preds.numpy(), smooth=True,
                    ignore_tokens=ignore_tokens,
                    ignore_all_tokens_after=ignore_all_tokens_after)[0]
                pbar.set_postfix(metrics)

                # shape: (batch_size, max_seq_len)
                preds = None
                # shape: (batch_size, 1, max_seq_len)
                references = None
                if eval_set is not None:
                    vloss = tf.constant(0, dtype=tf.float32)
                    n = tf.constant(0, dtype=tf.float32)
                    for X, y in eval_set:
                        enc_hidden = self.encoder.initialize_hidden_size(
                            y.shape[0])
                        vloss += self.forward_step(
                            X, y, enc_hidden, epoch+1, n+1, nottraining)
                        predictions, logprobas = self._decode_sequence(
                            X[0], X[1], X[2], X[3], nottraining)
                        if preds is None:
                            preds = predictions[:, 0]
                            references = y
                            references = tf.expand_dims(references, 1)
                        else:
                            refs = y
                            refs = tf.expand_dims(refs, 1)
                            references = tf.concat(
                                [references, refs], axis=0)

                            pred = predictions[:, 0]
                            preds = tf.concat(
                                [preds, pred], axis=0)
                        n += 1
                    metrics['val-loss'] = f"{vloss/n:.4f}"
                    metrics["val-bleu"] = compute_bleu(
                        references.numpy(), preds.numpy(),
                        ignore_tokens=ignore_tokens,
                        ignore_all_tokens_after=ignore_all_tokens_after)[0]
                    metrics["val-bleu-smooth"] = compute_bleu(
                        references.numpy(), preds.numpy(), smooth=True,
                        ignore_tokens=ignore_tokens,
                        ignore_all_tokens_after=ignore_all_tokens_after)[0]
                    pbar.set_postfix(metrics)
                ckpt_manager.save()

    def predict(self, dataset):
        ret_val = None
        logprobas = None
        for X, y in dataset:
            op, logproba = self._decode_sequence(X[0], X[1], X[2], X[3])
            if ret_val is None:
                ret_val = op
                logprobas = logproba
                continue
            ret_val = tf.concat([ret_val, op], 0)
            logprobas = tf.concat([logprobas, logproba], 0)
        return ret_val, logprobas

    def _load(self, save_loc):
        optimizer = tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM)
        ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, save_loc, max_to_keep=10)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        return optimizer, ckpt, ckpt_manager

    def load(self, save_loc):
        self._load(save_loc)
