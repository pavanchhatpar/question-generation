import tensorflow as tf
import argparse
import logging
from copynet_tf import Vocab

from text_gan import cfg, cfg_from_file
from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.features import GloVeReader, FastTextReader, NERTagger, PosTagger
from text_gan.models import CANPZ_Q

from text_gan.data.squad1_ca_qc import SQuAD_CA_QC
from text_gan.models import CANP_QC

# tf.debugging.set_log_device_placement(True)

MODELS = [
    "canpz-q",
    "canp-qc",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", choices=MODELS,
        required=True, dest="model",
        help="Select model to run predictions from")
    parser.add_argument(
        "--cfg", dest="cfg", type=str, help="Config YAML filepath",
        required=False, default=None)
    args = parser.parse_args()
    return args


def canpz_q():
    RNG_SEED = 11
    data = Squad1_CA_Q()
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = data.skip(1000).take(10000)\
        .batch(128).apply(to_gpu)
    val = data.take(1000).batch(128).apply(to_gpu)
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

    model = CANPZ_Q(vocab, ner, pos)
    loc = cfg.MODEL_SAVE
    # if os.path.exists(loc):
    #     shutil.rmtree(loc)
    model.fit(
        train, epochs=cfg.EPOCHS,
        save_loc=loc, eval_set=val)


def canp_qc():
    RNG_SEED = 11
    data = SQuAD_CA_QC()
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    train = data.take(20000).batch(128).apply(to_gpu)
    val = data.skip(20000).take(2000).batch(128).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(3)
        val = val.prefetch(2)
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
    loc = cfg.MODEL_SAVE
    model.fit(
        train, epochs=cfg.EPOCHS,
        save_loc=loc, eval_set=val, warm_start=True)


MODEL_METHODS = {
    "canpz-q": canpz_q,
    "canp-qc": canp_qc,
}


def main():
    args = parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    logging.basicConfig(
        level=cfg.LOG_LVL,
        filename=cfg.LOG_FILENAME,
        format='%(message)s')
    MODEL_METHODS[args.model]()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    # tf.debugging.set_log_device_placement(True)
    main()
