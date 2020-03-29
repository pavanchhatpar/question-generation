import tensorflow as tf
import argparse
import numpy as np
import ujson as json

from text_gan.data import QuestionContextPairs, CONFIG
from text_gan.models import QGAN, AttnGen

MODELS = [
    "qgen",
    "attn-qgen"
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
    word_emb_mat = np.array(json.load(open(CONFIG.WORDEMBMAT, "r")))
    qword_emb_mat = np.load(f"{CONFIG.QWORDEMBMAT}.npy")
    qword2idx = json.load(open(CONFIG.QWORD2IDX, "r"))
    idx2qword = np.full(CONFIG.QVOCAB_SIZE, "<UNK>", dtype='object')
    for word, idx in qword2idx.items():
        idx2qword[idx] = word
    word2idx = json.load(open(CONFIG.WORD2IDX, "r"))
    idx2word = np.full(len(word2idx), "<UNK>", dtype='object')
    for word, idx in word2idx.items():
        idx2word[idx] = word

    model = AttnGen(word_emb_mat, qword_emb_mat, qword2idx, idx2qword, CONFIG)
    model.model.compile(
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    model.load("/tf/data/attngen/model/model.tf")

    data = QuestionContextPairs.load(CONFIG.SAVELOC)
    train = data.train.take(10).batch(1)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)

    pred = model.predict(train)
    i = 0
    for X, y in train:
        context = idx2word[X[0]]
        context = filter(lambda w: w != '--NULL--', context)
        ques = map(lambda idx: idx2qword[idx], pred[i])
        print(f"Context:- {' '.join(context)}")
        print(f"Question:- {' '.join(ques)}")
        i += 1


def qgen():
    model = QGAN(CONFIG)
    model.plot_model()
    data = QuestionContextPairs.load(CONFIG.SAVELOC)

    train = data.train.take(10).batch(1)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)

    pred = model.predict(train)
    i = 0
    for X, y in train:
        c = model.idx2word[X[0]]
        print(f"Context:- {' '.join([w for w in c if w != '--NULL--'])}")
        print(f"Question:- {pred[i]}")
        i += 1


MODEL_METHODS = {
    "qgen": qgen,
    "attn-qgen": attn_qgen
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
