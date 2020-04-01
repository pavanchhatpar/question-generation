import tensorflow as tf
import numpy as np
import ujson as json
import argparse

from text_gan.data.qgen_data import QuestionContextPairs, CONFIG
from text_gan.data.qgen_ca_q import CA_QPair, CA_Qcfg
from text_gan.models import AttnGen, QGAN, CA_Q_AttnQGen, CAZ_Q_Attn

# tf.debugging.set_log_device_placement(True)

MODELS = [
    "qgen",
    "attn-qgen",
    "ca-q-attn-qgen",
    "caz-q-attn"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", choices=MODELS,
        required=True, dest="model",
        help="Select model to run predictions from")
    args = parser.parse_args()
    return args


def attn_qgen():
    data = QuestionContextPairs.load(CONFIG.SAVELOC)
    train = data.train.take(10000).batch(32)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
    word_emb_mat = np.array(json.load(open(CONFIG.WORDEMBMAT, "r")))
    qword_emb_mat = np.load(f"{CONFIG.QWORDEMBMAT}.npy")
    qword2idx = json.load(open(CONFIG.QWORD2IDX, "r"))
    idx2qword = np.full(CONFIG.QVOCAB_SIZE, "<UNK>", dtype='object')
    for word, idx in qword2idx.items():
        idx2qword[idx] = word
    model = AttnGen(word_emb_mat, qword_emb_mat, qword2idx, idx2qword, CONFIG)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "/tf/data/attn-qgen/checkpoint/ckpt.{epoch:03d}.tf",
            monitor='loss', verbose=1, save_weights_only=True)
    ]
    model.model.compile(
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    model.fit(train, epochs=100, callbacks=callbacks)
    model.save("/tf/data/attn-qgen/model/model.tf")


def qgen():
    data = QuestionContextPairs.load(CONFIG.SAVELOC)
    train = data.train.take(10000).batch(32)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
    model = QGAN(CONFIG, try_loading=False)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "/tf/data/qgen/checkpoint/ckpt.{epoch:03d}.tf",
            monitor='loss', verbose=1, save_weights_only=True)
    ]
    model.fit(train, epochs=100, callbacks=callbacks)
    model.save()


def ca_q_attn_qgen():
    data = CA_QPair.load()
    train = data.train.take(500).batch(8)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
    
    # with open(CA_Qcfg.CWORD2IDX, 'r') as f:
    #     cword2idx = json.load(f)
    # with open(CA_Qcfg.QWORD2IDX, 'r') as f:
    #     qword2idx = json.load(f)
    cidx2emb = np.load(CA_Qcfg.CIDX2EMB)
    qidx2emb = np.load(CA_Qcfg.QIDX2EMB)
    model = CA_Q_AttnQGen(cidx2emb, qidx2emb)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "/tf/data/ca-q-attn-qgen/checkpoint/ckpt.{epoch:03d}.tf",
            monitor='loss', save_weights_only=True)
    ]
    model.model.compile(
        tf.keras.optimizers.Adam(1e-3),
        'sparse_categorical_crossentropy'
    )
    try:
        model.fit(train, epochs=500, callbacks=callbacks)
        model.save("/tf/data/ca-q-attn-qgen/model/model.tf")
    except KeyboardInterrupt:
        print("Saving model trained so far")
        model.save("/tf/data/ca-q-attn-qgen/model/model.tf")


def caz_q_attn():
    RNG_SEED = 11
    data = CA_QPair.load()
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = data.skip(1000)\
        .shuffle(buffer_size=10000, seed=RNG_SEED)\
        .batch(8).apply(to_gpu)
    val = data.take(1000).batch(1000).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
        val = val.prefetch(1)

    # with open(CA_Qcfg.CWORD2IDX, 'r') as f:
    #     cword2idx = json.load(f)
    # with open(CA_Qcfg.QWORD2IDX, 'r') as f:
    #     qword2idx = json.load(f)

    cidx2emb = np.load(CA_Qcfg.CIDX2EMB)
    qidx2emb = np.load(CA_Qcfg.QIDX2EMB)
    model = CAZ_Q_Attn(cidx2emb, qidx2emb)
    model.fit(
        train, epochs=500, lr=1e-3,
        save_loc="/tf/data/caz-q-attn/", eval_set=val)


MODEL_METHODS = {
    "qgen": qgen,
    "attn-qgen": attn_qgen,
    "ca-q-attn-qgen": ca_q_attn_qgen,
    "caz-q-attn": caz_q_attn
}


def main():
    args = parse_args()
    MODEL_METHODS[args.model]()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    main()
