import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from .caz_q_decoder import CAZ_Q_Decoder
from .caz_q_encoder import CAZ_Q_Encoder
from ..data.qgen_ca_q import CA_Qcfg
from ..evaluation.bleu_score import bleu1, bleu4


class CAZ_Q_Attn:
    def __init__(
            self, word_emb_mat, qword_emb_mat):
        self.encoder = CAZ_Q_Encoder(word_emb_mat)
        self.decoder = CAZ_Q_Decoder(qword_emb_mat)

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
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=y_true)  # (batch, 1)

        # mask PAD token
        mask = tf.math.logical_not(tf.math.equal(y_true, CA_Qcfg.PAD_ID))
        mask = tf.cast(mask, dtype=cross_ent.dtype)
        cross_ent *= mask  # (batch, 1)

        logpy_z = -tf.reduce_sum(cross_ent, axis=1)
        logpz = CAZ_Q_Attn.log_normal_pdf(z, 0., 0.)
        logqz_y = CAZ_Q_Attn.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpy_z + logpz - logqz_y)

    @tf.function
    def train_step(self, X, y, enc_hidden):
        cidx, aidx, _ = X
        loss = 0

        with tf.GradientTape() as tape:
            s0, hd, mean, logvar, z = self.encoder(cidx, aidx, enc_hidden)

            START_TOKEN_ID = CA_Qcfg.START_ID
            BATCH_SIZE = cidx.shape[0]
            qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)

            for yi in range(y.shape[1]):
                ypred, s0 = self.decoder(qidx, s0, hd)

                loss += CAZ_Q_Attn.lossfn(y[:, yi], ypred, mean, logvar, z)

                qidx = tf.expand_dims(y[:, yi], 1)

            batch_loss = loss/y.shape[1]

            vars = (self.encoder.trainable_variables
                    + self.decoder.trainable_variables)

            gradients = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(gradients, vars))

        return batch_loss

    def fit(
            self, dataset, epochs, lr,
            save_loc, eval_set=None, eval_interval=10):
        self.optimizer = tf.keras.optimizers.Adam(lr)
        save_prefix = os.path.join(save_loc, "ckpt")
        ckpt_saver = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder)
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
                if epoch % eval_interval == 0:
                    train_bleu = self.evaluate(dataset)
                    metrics['train-bleu-1'] = train_bleu['bleu-1']
                    metrics['train-bleu-4'] = train_bleu['bleu-4']
                    if eval_set is not None:
                        eval_bleu = self.evaluate(eval_set)
                        metrics['eval-bleu-1'] = eval_bleu['bleu-1']
                        metrics['eval-bleu-4'] = eval_bleu['bleu-4']
                    pbar.set_postfix(metrics)
                ckpt_saver.save(file_prefix=save_prefix)

    @tf.function
    def _decode_sequence(self, cidx, aidx):
        # Encode the input as state vectors.
        enc_hidden = self.encoder.initialize_hidden_size(cidx.shape[0])
        s0, hd, mean, logvar, z = self.encoder(cidx, aidx, enc_hidden)

        # Start output sequence
        START_TOKEN_ID = CA_Qcfg.START_ID
        BATCH_SIZE = cidx.shape[0]
        qidx = tf.expand_dims([START_TOKEN_ID] * BATCH_SIZE, 1)  # (batch, 1)

        counter = 0
        op = qidx
        while counter < CA_Qcfg.QSEQ_LEN:
            y, st = self.decoder(qidx, s0, hd)
            # sampled_token_index = tf.multinomial(predictions, num_samples=1)
            sampled_token_index = tf.argmax(
                y, output_type=tf.int32, axis=2)  # (batch, 1)
            op = tf.concat([op, sampled_token_index], -1)
            qidx = sampled_token_index
            s0 = st
            counter += 1
            # if qidx == 3:  # EOS token
            #     break
        return op  # (batch, 20)

    def predict(self, dataset):
        ret_val = None
        for X, y in dataset:
            op = self._decode_sequence(X[0], X[1])
            if ret_val is None:
                ret_val = op
                continue
            ret_val = tf.concat([ret_val, op], 0)
        return ret_val  # (data_len, 20)

    def load(self, save_loc):
        optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt.restore(
            tf.train.latest_checkpoint(save_loc)).expect_partial()

    def get_numpy_scores(self, hypo, refs):
        fhypo = []
        frefs = []
        for hyp, ref in zip(hypo, refs):
            fhyp = []
            fref = []
            for h in hyp[1:]:
                if h == CA_Qcfg.END_ID:
                    break
                fhyp.append(h)
            fhypo.append(fhyp)
            for r in ref:
                if r == CA_Qcfg.END_ID:
                    break
                fref.append(r)
            frefs.append([fref])
        return [
            np.cast[np.float32](bleu1(frefs, fhypo)),
            np.cast[np.float32](bleu4(frefs, fhypo))]

    @tf.function
    def get_scores(self, hypo, refs):
        return tf.numpy_function(
            self.get_numpy_scores, [hypo, refs], (tf.float32, tf.float32))

    def evaluate(self, dataset):
        hypo = self.predict(dataset)
        refs = None
        for X, y in dataset:
            if refs is None:
                refs = y
            else:
                refs = tf.concat([refs, y], 0)
        bleu1, bleu4 = self.get_scores(hypo, refs)
        return {"bleu-1": bleu1, "bleu-4": bleu4}
