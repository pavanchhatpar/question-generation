import tensorflow as tf
import argparse
import numpy as np
import ujson as json
import logging

from text_gan.data.qgen_data import QuestionContextPairs, CONFIG
from text_gan.data.qgen_ca_q import CA_QPair, CA_Qcfg
from text_gan import cfg, cfg_from_file
from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.features import FastText, GloVe, NERTagger, PosTagger
from text_gan.models import QGAN, AttnGen, CA_Q_AttnQGen, CAZ_Q_Attn, CANPZ_Q

MODELS = [
    "qgen",
    "attn-qgen",
    "ca-q-attn-qgen",
    "caz-q-attn",
    "canpz-q",
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
    train = data.train.skip(500).take(100).batch(1)
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

    model.load('/tf/data/caz-q-attn-qgen/checkpoint')
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


def canpz_q():
    RNG_SEED = 11
    data = Squad1_CA_Q()
    data = data.train.shuffle(
        buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = data.skip(1000).take(5)\
        .shuffle(buffer_size=100, seed=RNG_SEED)\
        .batch(1).apply(to_gpu)
    val = data.take(1000).batch(10).apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
        val = val.prefetch(1)

    if cfg.EMBS_TYPE == 'glove':
        cembs = GloVe.load(cfg.EMBS_FILE, cfg.CSEQ_LEN, cfg.EMBS_CVOCAB)
        qembs = GloVe.load(
            cfg.EMBS_FILE, cfg.QSEQ_LEN, cfg.EMBS_QVOCAB, cembs.data)
    elif cfg.EMBS_TYPE == 'fasttext':
        cembs = FastText.load(cfg.EMBS_FILE, cfg.CSEQ_LEN, cfg.EMBS_CVOCAB)
        qembs = FastText.load(
            cfg.EMBS_FILE, cfg.QSEQ_LEN, cfg.EMBS_QVOCAB, cembs.data)
    else:
        raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
    ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
    pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)

    model = CANPZ_Q(cembs, ner, pos, qembs)
    model.load('/tf/data/canpz_q/')
    pred, attn_weights = model.predict(train)
    i = 0
    for X, y in train:
        context = cembs.inverse_transform(X[0].numpy())[0]
        answer = tf.reshape(X[0]*tf.cast(X[1], tf.int32), (-1,))
        ogques = qembs.inverse_transform(y.numpy())[0]
        ans = ''
        for ai in answer:
            if ai == 0:
                continue
            ans += cembs.inverse.get(ai.numpy(), cembs.UNK) + ' '
        context = filter(
            lambda w: w != cembs.PAD, context)
        ques = qembs.inverse_transform([pred[i].numpy()])[0]
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {X[1]}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Question:- {' '.join(ques)}")
        print(f"Attention Weights:- {attn_weights[i].numpy()}")
        print("")
        i += 1


MODEL_METHODS = {
    "qgen": qgen,
    "attn-qgen": attn_qgen,
    "ca-q-attn-qgen": ca_q_attn_qgen,
    "caz-q-attn": caz_q_attn,
    "canpz-q": canpz_q
}


def main():
    args = parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    logging.basicConfig(
        level=cfg.LOG_LVL,
        filename=cfg.LOG_FILENAME,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    MODEL_METHODS[args.model]()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    main()
