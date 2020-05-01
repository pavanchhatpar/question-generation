import tensorflow as tf
import argparse
import numpy as np
import logging

from text_gan import cfg, cfg_from_file, Vocab
from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.features import FastTextReader, GloVeReader, NERTagger, PosTagger
from text_gan.models import CANPZ_Q

MODELS = [
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
    pred, attn_weights = model.predict(val)
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
        ques = vocab.inverse_transform([pred[i].numpy()], "target")[0]
        try:
            ques = ques[:np.where(ques == embedding_reader.END)[0][0]]
        except:  # noqa
            pass
        attn_weight, idxs = tf.math.top_k(attn_weights[i][1:6], k=3)
        attn_tokens = tf.gather(X[0], idxs, axis=-1, batch_dims=0)[0]
        attn_tokens = vocab.inverse_transform(attn_tokens.numpy(), "source")
        print(f"Context:- {' '.join(context)}")
        print(f"Answer:- {ans}")
        print(f"OG Question:- {' '.join(ogques)}")
        print(f"Question:- {' '.join(ques[1:])}")
        print(
            f"Top attentive words for first 5 question tokens\n {attn_tokens}")
        print(f"Attention Weights:- {attn_weight}")
        print("")
        i += 1


MODEL_METHODS = {
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
