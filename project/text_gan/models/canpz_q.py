import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import logging

from .canpz_q_encoder import CANPZ_Q_Encoder
from .canpz_q_decoder import CANPZ_Q_Decoder
from ..config import cfg


class CANPZ_Q:
    def __init__(self, vocab, ner, pos, **kwargs):
        super(CANPZ_Q, self).__init__(**kwargs)
        self.encoder = CANPZ_Q_Encoder(vocab, ner, pos)
        self.decoder = CANPZ_Q_Decoder(vocab)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    @tf.function
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @staticmethod
    @tf.function
    def lossfn(y_true, y_pred, mean, logvar, z):
        y_true = tf.expand_dims(y_true, 1)  # (batch, 1)
        y_pred = tf.squeeze(y_pred)  # (batch, target_vocab_size)
        # shape: (batch, target_vocab_size)
        log_probs = tf.nn.log_softmax(y_pred)

        # shape: (batch, 1)
        cross_ent = -tf.gather(log_probs, y_true, axis=1, batch_dims=1)

        # cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=y_pred, labels=y_true)  # (batch, 1)

        # mask PAD token
        mask = tf.math.logical_not(tf.math.equal(y_true, cfg.PAD_ID))
        mask = tf.cast(mask, dtype=cross_ent.dtype)
        cross_ent *= mask  # (batch, 1)
        mask = tf.squeeze(mask)

        logpy_z = -tf.reduce_sum(cross_ent, axis=1)
        logpz = CANPZ_Q.log_normal_pdf(z, 0., 0.)*mask
        logqz_y = CANPZ_Q.log_normal_pdf(z, mean, logvar)*mask

        return -tf.reduce_mean(logpy_z + logpz - logqz_y)

    @tf.function
    def forward_step(self, X, y, enc_hidden, training=None):
        cidx, aidx, ner, pos = X
        loss = 0
        self.logger.debug("Running forward through network")

        s0, hd, mean, logvar, z, source_mask = self.encoder(
            cidx, aidx, ner, pos, enc_hidden, training)

        START_TOKEN_ID = cfg.START_ID
        BATCH_SIZE = cidx.shape[0]
        qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)

        for yi in range(1, y.shape[1]):
            ypred, s0, attn_weights = self.decoder(
                qidx, s0, hd, source_mask, training)
            if self.logger.getEffectiveLevel() == logging.DEBUG:
                # shape: (batch, source_seq_len)
                sq_attn_weights = tf.squeeze(attn_weights)
                k = 5
                samples = 3
                # shape: (samples, k) both
                max_attn, indices = tf.math.top_k(
                    sq_attn_weights[:samples], k=k)
                # shape: (samples, k)
                cidx_tokens = tf.gather(
                    cidx[:samples], indices, axis=1, batch_dims=1)
                self.logger.debug(
                    f"Attn for top {k} tokens in {samples} samples")
                self.logger.debug(f"context token ids: {cidx_tokens}")
                self.logger.debug(f"context posn: {indices}")
                self.logger.debug(f"attn values: {max_attn}")
                self.logger.debug(
                    f"used {qidx[:samples, 0]} to predict {y[:samples, yi]}")
                self.logger.debug(f"predicted{ypred[:samples, 0]}")
            loss += CANPZ_Q.lossfn(y[:, yi], ypred, mean, logvar, z)

            qidx = tf.expand_dims(y[:, yi], 1)

        # loss /= y.shape[1]

        return loss

    @tf.function
    def train_step(self, X, y, enc_hidden):
        with tf.GradientTape() as tape:
            loss = self.forward_step(X, y, enc_hidden, True)

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
        else:
            self.optimizer, ckpt_saver = self._load(save_loc)
        save_prefix = os.path.join(save_loc, "ckpt")
        for epoch in range(epochs):
            eloss = 0
            i = 0
            with tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for X, y in pbar.iterable:
                    enc_hidden = self.encoder.initialize_hidden_size(
                        y.shape[0])
                    bloss = self.train_step(X, y, enc_hidden)
                    pbar.update(1)
                    i += 1
                    eloss = (eloss*(i-1) + bloss)/i
                    metrics = {"train-loss": f"{eloss:.4f}"}
                    pbar.set_postfix(metrics)
                if eval_set is not None:
                    vloss = 0
                    n = 0
                    for X, y in eval_set:
                        enc_hidden = self.encoder.initialize_hidden_size(
                            y.shape[0])
                        vloss += self.forward_step(X, y, enc_hidden)
                        n += 1
                    metrics['val-loss'] = f"{vloss/n:.4f}"
                    pbar.set_postfix(metrics)
                ckpt_saver.save(file_prefix=save_prefix)

    @tf.function
    def _decode_sequence(self, cidx, aidx, ner, pos):
        # Encode the input as state vectors.
        enc_hidden = self.encoder.initialize_hidden_size(cidx.shape[0])
        s0, hd, mean, logvar, z, source_mask = self.encoder(
            cidx, aidx, ner, pos, enc_hidden)

        # Start output sequence
        START_TOKEN_ID = cfg.START_ID
        BATCH_SIZE = cidx.shape[0]
        qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)  # (batch, 1)

        counter = 0
        op = qidx
        attn = tf.zeros((BATCH_SIZE, 1, cfg.CSEQ_LEN))
        while counter < cfg.QSEQ_LEN:
            y, st, attn_weights = self.decoder(qidx, s0, hd, source_mask)
            # sampled_token_index = tf.multinomial(predictions, num_samples=1)
            sampled_token_index = tf.argmax(
                y, output_type=tf.int32, axis=2)  # (batch, 1)
            op = tf.concat([op, sampled_token_index], -1)
            attn = tf.concat([attn, attn_weights], -2)
            qidx = sampled_token_index
            s0 = st
            counter += 1
            # if qidx == cfg.END_ID:  # EOS token
            #     break
        return op, attn  # (batch, 20), (batch, 20, 250)

    def predict(self, dataset):
        ret_val = None
        attn_weights = None
        for X, y in dataset:
            op, attn = self._decode_sequence(X[0], X[1], X[2], X[3])
            if ret_val is None:
                ret_val = op
                attn_weights = attn
                continue
            ret_val = tf.concat([ret_val, op], 0)
            attn_weights = tf.concat([attn_weights, attn], 0)
        return ret_val, attn_weights  # (data_len, 20)

    def _load(self, save_loc):
        optimizer = tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM)
        ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt.restore(
            tf.train.latest_checkpoint(save_loc)).expect_partial()
        return optimizer, ckpt

    def load(self, save_loc):
        self._load(save_loc)
