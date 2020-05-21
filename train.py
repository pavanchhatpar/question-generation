import tensorflow as tf
from absl import logging, flags, app
from logging import Formatter
from copynet_tf import Vocab
from copynet_tf.loss import CopyNetLoss
from copynet_tf.metrics import BLEU
import os

from text_gan import cfg, cfg_from_file
from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.features import GloVeReader, FastTextReader, NERTagger, PosTagger
from text_gan.models import CANPZ_Q

from text_gan.data.squad1_ca_qc import SQuAD_CA_QC
from text_gan.models import CANP_QC

from text_gan.data.squad_ca_preqc import SQuAD_CA_PreQC
from text_gan.models import CANP_PreQC

# tf.debugging.set_log_device_placement(True)

FLAGS = flags.FLAGS


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
    train = data.take(cfg.TRAIN_SIZE).batch(cfg.BATCH_SIZE).apply(to_gpu)
    val = data.skip(cfg.TRAIN_SIZE).take(
        cfg.VAL_SIZE).batch(cfg.BATCH_SIZE).apply(to_gpu)
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


def canp_preqc():
    RNG_SEED = 11
    data = SQuAD_CA_PreQC()
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    train = data.take(cfg.TRAIN_SIZE).batch(
        cfg.BATCH_SIZE, drop_remainder=True).repeat(37).apply(to_gpu)
    val = data.skip(cfg.TRAIN_SIZE).take(
        cfg.VAL_SIZE).batch(cfg.BATCH_SIZE, drop_remainder=True).apply(to_gpu)
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

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        cfg.MODEL_SAVE+"/{epoch:02d}.tf", monitor='val_bleu',
        save_weights_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        FLAGS.log_dir, write_images=True)

    if cfg.STEPS_PER_EPOCH == -1:
        cfg.STEPS_PER_EPOCH = None

    _ = model.fit(
        train, epochs=cfg.EPOCHS, validation_data=val, shuffle=False,
        steps_per_epoch=cfg.STEPS_PER_EPOCH,
        callbacks=[
            ckpt,
            tensorboard
        ])


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

    if FLAGS.log_dir is not None:
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
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(
    #                 memory_limit=1024*10)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    # tf.debugging.set_log_device_placement(True)
    app.run(main)
