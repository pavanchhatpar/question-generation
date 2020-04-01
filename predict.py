import tensorflow as tf
import argparse
import numpy as np
import ujson as json

from text_gan.data.qgen_data import QuestionContextPairs, CONFIG
from text_gan.data.qgen_ca_q import CA_QPair, CA_Qcfg
from text_gan.models import QGAN, AttnGen, CA_Q_AttnQGen, CAZ_Q_Attn

MODELS = [
    "qgen",
    "attn-qgen",
    "ca-q-attn-qgen",
    "caz-q-attn",
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
    model.load("/tf/data/attn-qgen/model/model.tf")

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
        ogques = idx2qword[y]
        print(f"Context:- {' '.join(context)}")
        print(f"OG Question:- {' '.join(ogques)}")
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
        ogques = model.idx2qword[y]
        print(f"Context:- {' '.join([w for w in c if w != '--NULL--'])}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Question:- {pred[i]}")
        i += 1


def ca_q_attn_qgen():
    data = CA_QPair.load()
    train = data.train.take(10).batch(1)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)

    with open(CA_Qcfg.CWORD2IDX, 'r') as f:
        cword2idx = json.load(f)
    with open(CA_Qcfg.QWORD2IDX, 'r') as f:
        qword2idx = json.load(f)
    cidx2emb = np.load(CA_Qcfg.CIDX2EMB)
    qidx2emb = np.load(CA_Qcfg.QIDX2EMB)
    model = CA_Q_AttnQGen(cidx2emb, qidx2emb)
    model.model.compile(
        tf.keras.optimizers.Adam(1e-2),
        'sparse_categorical_crossentropy'
    )
    model.load("/tf/data/ca-q-attn-qgen/model/model.tf")
    cidx2word = np.full(len(cword2idx), CA_Qcfg.UNK_TOKEN, dtype='object')
    for token, idx in cword2idx.items():
        cidx2word[idx] = token

    qidx2word = np.full(len(qword2idx), CA_Qcfg.UNK_TOKEN, dtype='object')
    for token, idx in qword2idx.items():
        qidx2word[idx] = token

    pred = model.predict(train)
    i = 0
    for X, y in train:
        context = cidx2word[X[0]]
        answer = tf.reshape(X[0]*X[1], (-1,))
        ogques = qidx2word[y]
        ans = ''
        for ai in answer:
            if ai == 0:
                continue
            ans += cidx2word[ai] + ' '
        context = filter(
            lambda w: w != CA_Qcfg.PAD_TOKEN.decode('utf-8'), context)
        ques = map(lambda idx: qidx2word[idx], pred[i])
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Question:- {' '.join(ques)}")
        print("")
        i += 1


def caz_q_attn():
    data = CA_QPair.load()
    train = data.train.take(10).batch(1)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)

    with open(CA_Qcfg.CWORD2IDX, 'r') as f:
        cword2idx = json.load(f)
    with open(CA_Qcfg.QWORD2IDX, 'r') as f:
        qword2idx = json.load(f)
    cidx2emb = np.load(CA_Qcfg.CIDX2EMB)
    qidx2emb = np.load(CA_Qcfg.QIDX2EMB)
    model = CAZ_Q_Attn(cidx2emb, qidx2emb)
    cidx2word = np.full(len(cword2idx), CA_Qcfg.UNK_TOKEN, dtype='object')
    for token, idx in cword2idx.items():
        cidx2word[idx] = token

    qidx2word = np.full(len(qword2idx), CA_Qcfg.UNK_TOKEN, dtype='object')
    for token, idx in qword2idx.items():
        qidx2word[idx] = token

    model.load('/tf/data/caz-q-attn-qgen/checkpoint/')
    model.predict(train)
    pred = model.predict(train)
    i = 0
    for X, y in train:
        context = cidx2word[X[0]]
        answer = tf.reshape(X[0]*X[1], (-1,))
        ogques = qidx2word[y]
        ans = ''
        for ai in answer:
            if ai == 0:
                continue
            ans += cidx2word[ai] + ' '
        context = filter(
            lambda w: w != CA_Qcfg.PAD_TOKEN.decode('utf-8'), context)
        ques = map(lambda idx: qidx2word[idx], pred[i])
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Question:- {' '.join(ques)}")
        print("")
        i += 1
    print(model.evaluate(train))


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
