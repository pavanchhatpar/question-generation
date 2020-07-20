import tensorflow as tf
import numpy as np
from logging import Formatter
from absl import app, flags, logging
import json
import os
import en_core_web_sm

from copynet_tf import Vocab
from copynet_tf.loss import CopyNetLoss

from text_gan.utils import SQuADReader
from text_gan import cfg, cfg_from_file
from text_gan.features import FastTextReader, GloVeReader, NERTagger, PosTagger
from text_gan.models import CANP_QC


def substrSearch(ans, context):
    i = 0
    j = 0
    s = -1
    while i < len(context) and j < len(ans):
        if context[i].text == ans[j].text:
            if s == -1:
                s = i
            i += 1
            j += 1
        else:
            i += 1
            j = 0
            s = -1

    return s, j


def prepare_dataset(data, vocab, ner, pos):
    nlp = en_core_web_sm.load()

    context = map(lambda x: x['context'], data)
    qids = map(lambda x: x['qid'], data)
    answer = map(lambda x: x['answer'], data)

    context = nlp.pipe(context, batch_size=256, n_process=cfg.MAX_PARALLELISM)
    answer = nlp.pipe(answer, batch_size=256, n_process=cfg.MAX_PARALLELISM)

    final_context = []
    final_answer = []
    final_qids = []

    for cont, ans, qid in zip(
            context, answer, qids):
        ans_start, al = substrSearch(ans, cont)
        ans_start += 1
        if ans_start == -1 or ans_start + al >= 250:
            continue
        final_context.append(cont)
        final_qids.append(qid)
        ans = np.zeros(cfg.CSEQ_LEN, dtype=np.uint8)
        ans[ans_start] = 1
        ans[ans_start+1:ans_start+al] = 2
        final_answer.append(ans)

    ciss = vocab.transform(final_context, "source")
    cits = vocab.transform(final_context, "target", cfg.CSEQ_LEN)
    ners = ner.transform(final_context)
    poss = pos.transform(final_context)

    cseq = cfg.CSEQ_LEN

    def gen():
        for cis, cit, ans, ner1, pos1 in zip(
                ciss, cits, final_answer, ners, poss):
            yield ((cis, cit, ans, ner1, pos1),)

    X = tf.data.Dataset.from_generator(
        gen,
        ((tf.int32, tf.int32, tf.uint8, tf.uint8, tf.uint8),),
        ((
            tf.TensorShape([cseq]), tf.TensorShape([cseq]),
            tf.TensorShape([cseq]), tf.TensorShape([cseq]),
            tf.TensorShape([cseq])
        ),)
    )

    return X, final_qids


def idx2str(pred_y, X, vocab):
    ret = []
    vocab_len = vocab.get_vocab_size("target")
    for idx in pred_y:
        if idx < vocab_len:
            ret.append(vocab.get_token_text(idx, "target"))
        else:
            ret.append(vocab.get_token_text(X[idx-vocab_len], "source"))
    return ret


def canp_qc():
    reader = SQuADReader()
    data = reader.flatten_parsed(reader.parse(FLAGS.set, qids=True), qids=True)
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

    X, qids = prepare_dataset(data, vocab, ner, pos)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    X = X.batch(128, drop_remainder=True).apply(to_gpu)
    counter = 0
    for x in X:
        counter += 1
    with tf.device("/gpu:0"):
        X = X.prefetch(3)

    print(f"***Dataset ready {counter} batches***")

    model = CANP_QC(vocab, ner, pos)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM),
        loss=CopyNetLoss(),
        metrics=[
            # BLEU(ignore_tokens=[0, 2, 3], ignore_all_tokens_after=3),
            # BLEU(ignore_tokens=[0, 2, 3], ignore_all_tokens_after=3,
            #      name='bleu-smooth', smooth=True)
        ]
    )
    filename = tf.train.latest_checkpoint(cfg.MODEL_SAVE)
    model.load_weights(filename)

    print("***Model ready***")

    out = model.predict(X)
    preds = out['predictions']

    print("***Predictions ready***")
    out = {}
    for x, qid, pred in zip(X.unbatch(), qids, preds):
        cis, cit, answer, ner, pos = x[0]
        # context = vocab.inverse_transform([cis.numpy()], "source")[0]
        # context = filter(
        #     lambda w: w != embedding_reader.PAD, context)
        # context = " ".join(context)
        # ans = ''
        # for j, ai in enumerate(answer):
        #     if ai == 0:
        #         continue
        #     ans += vocab.get_token_text(cis[j].numpy(), "source") + ' '
        questions = []
        for j in range(3):
            p = idx2str(pred[j], cis.numpy(), vocab)
            questions.append(" ".join(p))
        out[qid] = questions

    with open(FLAGS.out, "w") as fp:
        json.dump(out, fp)


MODEL_METHODS = {
    "canp-qc": canp_qc,
}

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", None, "Config YAML filepath")
flags.DEFINE_string("set", None, "train/dev set")
flags.DEFINE_string("out", None, "Output filename")


def main(_):
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

    if FLAGS.set is None or FLAGS.set not in ['train', 'dev']:
        raise ValueError("Choose a set, train/ dev")

    FLAGS.set = (
        cfg.RAW_TRAIN_SAVE if FLAGS.set == 'train' else cfg.RAW_DEV_SAVE)

    if FLAGS.out is None:
        raise ValueError("Give an output filename")

    MODEL_METHODS[cfg.MODEL]()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    app.run(main)
