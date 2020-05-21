import tensorflow as tf
from logging import Formatter
import numpy as np
from absl import logging, flags, app
from copynet_tf import Vocab
from copynet_tf.loss import CopyNetLoss
from copynet_tf.metrics import BLEU
import os

from text_gan import cfg, cfg_from_file
from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.features import FastTextReader, GloVeReader, NERTagger, PosTagger
from text_gan.models import CANPZ_Q

from text_gan.data.squad1_ca_qc import SQuAD_CA_QC
from text_gan.models import CANP_QC

from text_gan.data.squad_ca_preqc import SQuAD_CA_PreQC
from text_gan.models import CANP_PreQC

FLAGS = flags.FLAGS


def canpz_q():
    RNG_SEED = 11
    data = Squad1_CA_Q()
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = data.skip(1000).take(10)\
        .batch(10).apply(to_gpu)
    val = data.take(10).batch(10).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
        val = val.prefetch(1)

    if cfg.EMBS_TYPE == 'glove':
        embedding_reader = GloVeReader()
    elif cfg.EMBS_TYPE == 'fasttext':
        embedding_reader = FastTextReader()
    else:
        raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
    vocab = Vocab.load(
        embedding_reader.START,
        embedding_reader.END,
        embedding_reader.PAD,
        embedding_reader.UNK,
        cfg.CSEQ_LEN,
        cfg.QSEQ_LEN,
        cfg.VOCAB_SAVE
    )
    ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
    pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)

    model = CANPZ_Q(vocab, ner, pos)
    model.load(cfg.MODEL_SAVE)
    pred, logprobas = model.predict(val)
    i = 0
    for X, y in val.unbatch().batch(1):
        context = vocab.inverse_transform(X[0].numpy(), "source")[0]
        answer = tf.reshape(X[0]*tf.cast(X[1], tf.int32), (-1,))
        ogques = vocab.inverse_transform(y.numpy(), "target")[0]
        ans = ''
        for ai in answer:
            if ai == 0:
                continue
            ans += vocab.get_token_text(ai.numpy(), "source") + ' '
        context = filter(
            lambda w: w != embedding_reader.PAD, context)
        try:
            ogques = ogques[:np.where(ogques == embedding_reader.END)[0][0]]
        except:  # noqa
            pass
        ques = vocab.inverse_transform(pred[i].numpy(), "target")
        # try:
        #     ques = ques[:np.where(ques == embedding_reader.END)[0][0]]
        # except:  # noqa
        #     pass
        # attn_weight, idxs = tf.math.top_k(attn_weights[i][1:6], k=3)
        # attn_tokens = tf.gather(X[0], idxs, axis=-1, batch_dims=0)[0]
        # attn_tokens = vocab.inverse_transform(attn_tokens.numpy(), "source")
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Top Questions:-\n{[' '.join(q) for q in ques]}")
        # print(f"Log probs:- {logprobas[i]}")
        # print(
        #     f"Top attentive words for first 5 question tokens\n {attn_tokens}")
        # print(f"Attention Weights:- {attn_weight}")
        print("")
        i += 1


def canp_qc():
    RNG_SEED = 11
    data = SQuAD_CA_QC()
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    train = data.take(10).batch(128).apply(to_gpu)
    val = data.skip(cfg.TRAIN_SIZE).take(10).batch(128).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(1)
        val = val.prefetch(1)
    if cfg.EMBS_TYPE == 'glove':
        embedding_reader = GloVeReader()
    elif cfg.EMBS_TYPE == 'fasttext':
        embedding_reader = FastTextReader()
    else:
        raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
    vocab = Vocab.load(
        embedding_reader.START,
        embedding_reader.END,
        embedding_reader.PAD,
        embedding_reader.UNK,
        cfg.CSEQ_LEN,
        cfg.QSEQ_LEN,
        cfg.VOCAB_SAVE
    )
    ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
    pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)

    model = CANP_QC(vocab, ner, pos)
    model.load(cfg.MODEL_SAVE)
    pred, logprobas = model.predict(val)
    i = 0
    for X, y in val.unbatch():
        cis, cit, answer, ner, pos = X
        qit, qis = y
        context = vocab.inverse_transform([cis.numpy()], "source")[0]
        ogques = vocab.inverse_transform([qit.numpy()], "target")[0]
        ans = ''
        for j, ai in enumerate(answer):
            if ai == 0:
                continue
            ans += vocab.get_token_text(cis[j].numpy(), "source") + ' '
        context = filter(
            lambda w: w != embedding_reader.PAD, context)
        # try:
        #     ogques = ogques[:np.where(ogques == embedding_reader.END)[0][0]]
        # except:  # noqa
        #     pass
        # ques = vocab.inverse_transform(pred[i].numpy(), "target")
        # try:
        #     ques = ques[:np.where(ques == embedding_reader.END)[0][0]]
        # except:  # noqa
        #     pass
        # attn_weight, idxs = tf.math.top_k(attn_weights[i][1:6], k=3)
        # attn_tokens = tf.gather(X[0], idxs, axis=-1, batch_dims=0)[0]
        # attn_tokens = vocab.inverse_transform(attn_tokens.numpy(), "source")
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"OG Question:- {' '.join(ogques)}")

        print(f"Top Questions:")
        for j in range(10):
            p = idx2str(pred[i][j].numpy(), cis.numpy(), vocab)
            print(f"Predicted: {' '.join(p)}\t"
                  f"Proba: {tf.exp(logprobas[i][j])}")
        # print(f"Log probs:- {logprobas[i]}")
        # print(
        #     f"Top attentive words for first 5 question tokens\n {attn_tokens}")
        # print(f"Attention Weights:- {attn_weight}")
        print("")
        i += 1


def canp_preqc():
    RNG_SEED = 11
    data = SQuAD_CA_PreQC()
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    train = data.take(10).batch(10, drop_remainder=True).apply(to_gpu)
    val = data.skip(cfg.TRAIN_SIZE).skip(cfg.VAL_SIZE).take(10).batch(
        10, drop_remainder=True).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(1)
        val = val.prefetch(1)
    if cfg.EMBS_TYPE == 'glove':
        embedding_reader = GloVeReader()
    elif cfg.EMBS_TYPE == 'fasttext':
        embedding_reader = FastTextReader()
    else:
        raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
    vocab = Vocab.load(
        embedding_reader.START,
        embedding_reader.END,
        embedding_reader.PAD,
        embedding_reader.UNK,
        cfg.CSEQ_LEN,
        cfg.QSEQ_LEN,
        cfg.VOCAB_SAVE
    )
    ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
    pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)

    model = CANP_PreQC(vocab, ner, pos)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM),
        loss=CopyNetLoss(),
        metrics=[
            BLEU(ignore_tokens=[0, 2, 3], ignore_all_tokens_after=3),
            BLEU(ignore_tokens=[0, 2, 3], ignore_all_tokens_after=3,
                 name='bleu-smooth', smooth=True)
        ]
    )
    filename = tf.train.latest_checkpoint(cfg.MODEL_SAVE)
    model.load_weights(filename)
    out = model.predict(val)
    pred, logprobas = out['predictions'], out['predicted_probas']
    i = 0
    for X, y in val.unbatch():
        cis, cit, answer, ner, pos, preq = X
        qit, qis = y
        context = vocab.inverse_transform([cis.numpy()], "source")[0]
        ogques = vocab.inverse_transform([qit.numpy()], "target")[0]
        ogpref = vocab.inverse_transform([preq.numpy()], "target")[0]
        ans = ''
        for j, ai in enumerate(answer):
            if ai == 0:
                continue
            ans += vocab.get_token_text(cis[j].numpy(), "source") + ' '
        context = filter(
            lambda w: w != embedding_reader.PAD, context)
        # try:
        #     ogques = ogques[:np.where(ogques == embedding_reader.END)[0][0]]
        # except:  # noqa
        #     pass
        # ques = vocab.inverse_transform(pred[i].numpy(), "target")
        # try:
        #     ques = ques[:np.where(ques == embedding_reader.END)[0][0]]
        # except:  # noqa
        #     pass
        # attn_weight, idxs = tf.math.top_k(attn_weights[i][1:6], k=3)
        # attn_tokens = tf.gather(X[0], idxs, axis=-1, batch_dims=0)[0]
        # attn_tokens = vocab.inverse_transform(attn_tokens.numpy(), "source")
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"Ques prefix:- {' '.join(ogpref)}")
        print(f"OG Suffix:- {' '.join(ogques)}")

        print(f"Top Suffixes:")
        for j in range(10):
            p = idx2str(pred[i][j], cis.numpy(), vocab)
            print(f"Predicted: {' '.join(p)}\t"
                  f"Proba: {tf.exp(logprobas[i][j])}")
        # print(f"Log probs:- {logprobas[i]}")
        # print(
        #     f"Top attentive words for first 5 question tokens\n {attn_tokens}")
        # print(f"Attention Weights:- {attn_weight}")
        print("")
        i += 1


def idx2str(pred_y, X, vocab):
    ret = []
    vocab_len = vocab.get_vocab_size("target")
    for idx in pred_y:
        if idx < vocab_len:
            ret.append(vocab.get_token_text(idx, "target"))
        else:
            ret.append(vocab.get_token_text(X[idx-vocab_len], "source"))
    return ret


MODEL_METHODS = {
    "canpz-q": canpz_q,
    "canp-qc": canp_qc,
    "canp-preqc": canp_preqc,
}

flags.DEFINE_string("cfg", None, "Config YAML filepath")


def main(argv):
    del argv

    if FLAGS.cfg is not None:
        cfg_from_file(FLAGS.cfg)

    if FLAGS.log_dir is not None and FLAGS.log_dir != "":
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        if not os.path.isdir(FLAGS.log_dir):
            raise ValueError(f"{FLAGS.log_dir} should be a directory!")
        logging.get_absl_handler().use_absl_log_file()

    logging.get_absl_handler().setFormatter(
        Formatter(fmt="%(levelname)s:%(message)s"))

    MODEL_METHODS[cfg.MODEL]()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    app.run(main)
