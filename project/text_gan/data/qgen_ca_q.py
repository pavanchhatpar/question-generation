from read_only_class_attributes import read_only
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
import numpy as np
import ujson as json
import os
import pandas as pd
from collections import defaultdict

from .squad2 import Squad2  # noqa


@read_only('*')
class _CA_Qcfg:
    LOC = '/tf/data/processed'
    GLOVE_EMBS = '/tf/data/embeddings/glove.840B.300d.txt'
    EMBS_DIM = 300

    UNK_TOKEN = b'UNKNOWN'
    PAD_TOKEN = b'PAD'
    START_TOKEN = b'<S>'
    END_TOKEN = b'EOS'

    QVOCAB_SIZE = 5000
    QWORD2IDX = '/tf/data/embeddings/qword2idx.json'
    QIDX2EMB = '/tf/data/embeddings/qidx2emb.npy'
    QSEQ_LEN = 20  # rounding up 95th percentile

    CWORD2IDX = '/tf/data/embeddings/cword2idx.json'
    CIDX2EMB = '/tf/data/embeddings/cidx2emb.npy'
    CSEQ_LEN = 250  # rounding up 95th percentile


CA_Qcfg = _CA_Qcfg()


class CA_QPair:
    def __init__(self, create=True):
        if create:
            self.tokenizer = text.UnicodeScriptTokenizer()
            train = tfds.load(
                "squad2", data_dir="/tf/data/tf_data", split='train')
            val = tfds.load(
                "squad2", data_dir="/tf/data/tf_data", split='validation')
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            self.train = train.map(self.tokenize, num_parallel_calls=AUTOTUNE)
            self.val = val.map(self.tokenize, num_parallel_calls=AUTOTUNE)
            self.cword2idx, self.cidx2emb, self.qword2idx, self.qidx2emb =\
                self.generate_embeddings()
            self.cword2idx = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(self.cword2idx.keys()), list(self.cword2idx.values()),
                    key_dtype=tf.string, value_dtype=tf.int32
                ), default_value=tf.constant(1, dtype=tf.int32)
            )
            self.qword2idx = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(self.qword2idx.keys()), list(self.qword2idx.values()),
                    key_dtype=tf.string, value_dtype=tf.int32
                ), default_value=tf.constant(1, dtype=tf.int32)
            )
            self.train = self.train.map(
                self.make_example, num_parallel_calls=AUTOTUNE)
            self.val = self.val.map(
                self.make_example, num_parallel_calls=AUTOTUNE)

    def save(self):
        if not os.path.exists(CA_Qcfg.LOC):
            os.makedirs(CA_Qcfg.LOC)
        if not os.path.isdir(CA_Qcfg.LOC):
            raise ValueError(f"{CA_Qcfg.LOC} should be a directory!")
        print("******** Saving Validation set ********")
        fname = os.path.join(CA_Qcfg.LOC, "CA_Q.val.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(self.val)

        print("******** Saving Training set ********")
        fname = os.path.join(CA_Qcfg.LOC, "CA_Q.train.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(self.train)
        print("******** Finished saving dataset ********")

    @staticmethod
    def parse_ex(example_proto):
        feature_description = {
            'cidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'aidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qidx': tf.io.FixedLenFeature([], tf.string, default_value='')
        }
        example = tf.io.parse_single_example(
            example_proto, feature_description)
        cidx = tf.io.parse_tensor(example['cidx'], out_type=tf.int32)
        cidx.set_shape([CA_Qcfg.CSEQ_LEN, ])
        qidx = tf.io.parse_tensor(example['qidx'], out_type=tf.int32)
        qidx.set_shape([CA_Qcfg.QSEQ_LEN+1, ])
        aidx = tf.io.parse_tensor(example['aidx'], out_type=tf.int32)
        aidx.set_shape([CA_Qcfg.CSEQ_LEN, ])
        return ((cidx, aidx, qidx[:-1]), qidx[1:])

    @classmethod
    def load(cls):
        if not os.path.exists(CA_Qcfg.LOC) or not os.path.isdir(CA_Qcfg.LOC):
            raise ValueError(f"{CA_Qcfg.LOC} should be a directory!")
        TRAIN = os.path.join(CA_Qcfg.LOC, "CA_Q.train.tfrecord")
        VAL = os.path.join(CA_Qcfg.LOC, "CA_Q.val.tfrecord")
        if not os.path.exists(TRAIN) or not os.path.exists(VAL):
            raise ValueError(f"Dataset not present inside {CA_Qcfg.LOC}!")
        train = tf.data.TFRecordDataset([TRAIN], compression_type='ZLIB')
        val = tf.data.TFRecordDataset([VAL], compression_type='ZLIB')
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train = train.map(cls.parse_ex, num_parallel_calls=AUTOTUNE)
        val = val.map(cls.parse_ex, num_parallel_calls=AUTOTUNE)
        inst = cls(create=False)
        inst.train = train
        inst.val = val
        return inst

    def tokenize(self, raw):
        context = self.tokenizer.tokenize(raw['context'])
        question = self.tokenizer.tokenize(raw['question'])

        if len(raw['answers']['text']) > 0:
            answer = self.tokenizer.tokenize(raw['answers']['text'][0])
        else:
            answer = self.tokenizer.tokenize(b'')

        return {
            "context": context,
            "question": question,
            "answer": answer
        }

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize(self, cidx, qidx, aidx):
        cidx = tf.io.serialize_tensor(cidx)
        qidx = tf.io.serialize_tensor(qidx)
        aidx = tf.io.serialize_tensor(aidx)
        feature = {
            "cidx": self._bytes_feature(cidx),
            "aidx": self._bytes_feature(aidx),
            "qidx": self._bytes_feature(qidx),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def mark_answer(self, cidx, aidx):
        cidx = cidx.numpy()
        aidx = aidx.numpy()
        if aidx.shape[0] == 0:
            return np.zeros(cidx.shape, dtype=np.int32)
        size = aidx.shape[0]
        shape = cidx.shape[:-1] + (cidx.shape[-1] - size + 1, size)
        strides = cidx.strides + (cidx.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(
            cidx, shape=shape, strides=strides)
        answer = np.all(windows == aidx, axis=1)
        aidx = np.zeros(cidx.shape, dtype=np.int32)
        if answer.nonzero()[0].shape[0] != 0:
            start_index = answer.nonzero()[0][0]
            for i in range(size):
                aidx[start_index+i] = 1
        return aidx

    def make_example(self, tokenized):
        cidx = self.cword2idx.lookup(tokenized['context'])
        aidx = self.cword2idx.lookup(tokenized['answer'])
        qidx = self.qword2idx.lookup(tokenized['question'])
        cidx = tf.py_function(
            self.py_pad_context,
            inp=[cidx], Tout=tf.int32
        )
        cidx.set_shape([CA_Qcfg.CSEQ_LEN, ])
        qidx = tf.py_function(
            self.py_pad_question,
            inp=[qidx], Tout=tf.int32
        )
        qidx.set_shape([CA_Qcfg.QSEQ_LEN+1, ])

        aidx = tf.py_function(
            self.mark_answer,
            [cidx, aidx],
            tf.int32)
        aidx.set_shape([CA_Qcfg.CSEQ_LEN, ])

        # serialize
        serialized = tf.py_function(
            self.serialize,
            [cidx, qidx, aidx],
            tf.string)
        return tf.reshape(serialized, ())

    def py_pad_context(self, tokens):
        tokens = tokens.numpy()
        if tokens.shape[0] < CA_Qcfg.CSEQ_LEN:
            rem = np.zeros(
                CA_Qcfg.CSEQ_LEN - tokens.shape[0], dtype=np.int32)
            return np.concatenate([tokens, rem])
        else:
            return tokens[:CA_Qcfg.CSEQ_LEN]

    def py_pad_question(self, tokens):
        tokens = tokens.numpy()
        if tokens.shape[0] < CA_Qcfg.QSEQ_LEN - 1:
            rem = np.zeros(
                CA_Qcfg.QSEQ_LEN - 1 - tokens.shape[0], dtype=np.int32)
            return np.concatenate(
                [np.array([2]), tokens,  np.array([3]), rem])
        else:
            return np.concatenate([
                np.array([2]),
                tokens[:CA_Qcfg.QSEQ_LEN - 1], np.array([3])])

    def generate_embeddings(self):
        if not os.path.exists(CA_Qcfg.QWORD2IDX):
            word2embs = {}
            with open(CA_Qcfg.GLOVE_EMBS, "r") as f:
                line = f.readline()
                while len(line) != 0:
                    word_vec = line.split(' ')
                    word = word_vec[0]
                    vec = np.array(word_vec[1:], dtype=np.float32)
                    word2embs[word.encode('utf-8')] = vec
                    line = f.readline()
            print(f"{len(word2embs)} words in GloVe")
            cvocab = defaultdict(lambda: 0)
            qvocab = defaultdict(lambda: 0)
            for example in self.train.as_numpy_iterator():
                for i, token in enumerate(example['context']):
                    if i >= CA_Qcfg.CSEQ_LEN:
                        break
                    cvocab[token] += 1
                for i, token in enumerate(example['question']):
                    if i >= CA_Qcfg.QSEQ_LEN:
                        break
                    qvocab[token] += 1

            cdf = pd.DataFrame(
                {'token': list(cvocab.keys()), 'n': list(cvocab.values())})\
                .sort_values('n', ascending=False)

            qdf = pd.DataFrame(
                {'token': list(qvocab.keys()), 'n': list(qvocab.values())})\
                .sort_values('n', ascending=False)

            cword2idx = {
                CA_Qcfg.PAD_TOKEN: 0,
                CA_Qcfg.UNK_TOKEN: 1,
                CA_Qcfg.START_TOKEN: 2,
                CA_Qcfg.END_TOKEN: 3
            }
            cidx2emb = [
                word2embs[CA_Qcfg.PAD_TOKEN],
                word2embs[CA_Qcfg.UNK_TOKEN],
                word2embs[CA_Qcfg.START_TOKEN],
                word2embs[CA_Qcfg.END_TOKEN]
            ]
            oov = 0
            i = 4
            for token in cdf[cdf.n > 10].token:
                if token in word2embs:
                    cword2idx[token] = i
                    cidx2emb.append(word2embs[token])
                    i += 1
                else:
                    oov += 1
            cidx2emb = np.array(cidx2emb, dtype=np.float32)
            np.save(CA_Qcfg.CIDX2EMB, cidx2emb)
            with open(CA_Qcfg.CWORD2IDX, 'w') as f:
                json.dump(cword2idx, f)
            print(f"{oov} context tokens not in GloVe")

            qword2idx = {
                CA_Qcfg.PAD_TOKEN: 0,
                CA_Qcfg.UNK_TOKEN: 1,
                CA_Qcfg.START_TOKEN: 2,
                CA_Qcfg.END_TOKEN: 3
            }
            qidx2emb = [
                word2embs[CA_Qcfg.PAD_TOKEN],
                word2embs[CA_Qcfg.UNK_TOKEN],
                word2embs[CA_Qcfg.START_TOKEN],
                word2embs[CA_Qcfg.END_TOKEN]
            ]
            i = 4
            oov = 0
            for token in qdf.token:
                if i >= CA_Qcfg.QVOCAB_SIZE:
                    break
                if token in word2embs:
                    qword2idx[token] = i
                    qidx2emb.append(word2embs[token])
                    i += 1
                else:
                    oov += 1
            qidx2emb = np.array(qidx2emb, dtype=np.float32)
            np.save(CA_Qcfg.QIDX2EMB, qidx2emb)
            with open(CA_Qcfg.QWORD2IDX, 'w') as f:
                json.dump(qword2idx, f)
            msg = f"{oov} oov tokens in top {CA_Qcfg.QVOCAB_SIZE}"\
                + " question tokens present"
            print(msg)
        else:
            with open(CA_Qcfg.CWORD2IDX, "r") as f:
                cword2idx = json.load(f)
            with open(CA_Qcfg.QWORD2IDX, "r") as f:
                qword2idx = json.load(f)
            cidx2emb = np.load(CA_Qcfg.CIDX2EMB)
            qidx2emb = np.load(CA_Qcfg.QIDX2EMB)

        return cword2idx, cidx2emb, qword2idx, qidx2emb
