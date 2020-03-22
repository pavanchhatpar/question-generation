import tensorflow_datasets as tfds
from .squad2 import Squad2  # noqa
import tensorflow_text as text
import tensorflow as tf
import numpy as np
import ujson as json
import os
from read_only_class_attributes import read_only


@read_only('*')
class _Config:
    WORD2IDX = "/tf/data/squad/word2idx.json"
    QWORD2IDX = "/tf/data/squad/qword2idx.json"
    DISCOURSEWORDS = "/tf/data/lexicon_rst_pdtb"
    WORDEMBMAT = "/tf/data/squad/word_emb.json"
    SAVELOC = "/tf/data/processed"
    MODELSAVELOC = "/tf/data/model.tf"
    MAX_CONTEXT_LEN = 256  # closest power of 2 from 95 %tile length
    MAX_QLEN = 16  # closest power of 2 from 95 %tile length
    EMBS_DIM = 300
    QVOCAB_SIZE = 5000
    LATENT_DIM = 32


CONFIG = _Config()


class _Tokenizer:
    def __init__(self):
        self.tokenizer = text.UnicodeScriptTokenizer()

    def tokenize(self, s):
        ret = self.tokenizer.tokenize(s).numpy()
        for i, r in enumerate(ret):
            ret[i] = r.decode("utf-8")
        return ret


class _DiscourseMapperTreeNode:
    def __init__(self):
        self.tokens = {}
        self.end = False

    def match(self, tokens, j):
        curr_node = self
        ret = 0
        last_found = 0
        for i in range(j, len(tokens)-j):
            if tokens[i] not in curr_node.tokens and curr_node.end:
                return ret
            try:
                curr_node = curr_node.tokens[tokens[i]]
                ret += 1
                if curr_node.end:
                    last_found = ret
            except KeyError:
                return last_found
        return 0


class QuestionContextPairs:
    def __init__(self, config, create=True):
        if create:
            self.config = config
            self.tokenizer = text.UnicodeScriptTokenizer()
            rawidx = json.load(open(self.config.WORD2IDX, "r"))
            word2idx = {}
            for k, v in rawidx.items():
                word2idx[k.encode("utf-8")] = v

            qrawidx = json.load(open(self.config.QWORD2IDX, "r"))
            qword2idx = {}
            for k, v in qrawidx.items():
                qword2idx[k.encode("utf-8")] = v
            self.qword2idx0 = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(qword2idx.keys()), list(qword2idx.values()),
                    key_dtype=tf.string, value_dtype=tf.int32),
                default_value=tf.constant(0, dtype=tf.int32))

            self.word2idx0 = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(word2idx.keys()), list(word2idx.values()),
                    key_dtype=tf.string, value_dtype=tf.int32),
                default_value=tf.constant(0, dtype=tf.int32))

            self.discourse_mapper = _DiscourseMapperTreeNode()
            with open(self.config.DISCOURSEWORDS, "r") as f:
                for discourse_marker in f:
                    tokens = discourse_marker.strip().split(' ')
                    curr_node = self.discourse_mapper
                    for i, token in enumerate(tokens):
                        token = token.encode("utf-8")
                        if token not in curr_node.tokens:
                            curr_node.tokens[token] = _DiscourseMapperTreeNode()  # noqa
                        curr_node = curr_node.tokens[token]
                        if i == len(tokens) - 1:
                            curr_node.end = True
            og_train = tfds.load(
                "squad2", data_dir="/tf/data/tf_data", split='train')
            og_val = tfds.load(
                "squad2", data_dir="/tf/data/tf_data", split='validation')
            self.train = og_train.map(self.mapper1, num_parallel_calls=-1)\
                .map(self.mapper2, num_parallel_calls=-1)
            self.val = og_val.map(self.mapper1, num_parallel_calls=-1)\
                .map(self.mapper2, num_parallel_calls=-1)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def py_discourse_marker_mapper(self, tokens):
        tokens = tokens.numpy()
        ret = np.zeros(self.config.MAX_CONTEXT_LEN, dtype=np.uint8)
        i = 0
        while i < len(tokens):
            if i >= self.config.MAX_CONTEXT_LEN:
                break
            discourse_len = self.discourse_mapper.match(tokens, i)
            for j in range(discourse_len):
                if i >= self.config.MAX_CONTEXT_LEN:
                    break
                ret[i] = 1
                i += 1
            if discourse_len == 0:
                i += 1
        return ret

    def mapper1(self, raw):
        return {
            "context": self.tokenizer.tokenize(raw['context']),
            "question": self.tokenizer.tokenize(raw['question'])
        }

    def py_pad_context(self, tokens):
        tokens = tokens.numpy()
        if tokens.shape[0] < self.config.MAX_CONTEXT_LEN:
            rem = np.zeros(
                self.config.MAX_CONTEXT_LEN - tokens.shape[0], dtype=np.int32)
            return np.concatenate([tokens, rem])
        else:
            return tokens[:self.config.MAX_CONTEXT_LEN]

    def py_pad_question(self, tokens):
        tokens = tokens.numpy()
        if tokens.shape[0] < self.config.MAX_QLEN - 1:
            rem = np.zeros(
                self.config.MAX_QLEN - 1 - tokens.shape[0], dtype=np.int32)
            return np.concatenate(
                [np.array([4998]), tokens,  np.array([4999]), rem])
        else:
            return np.concatenate([
                np.array([4998]),
                tokens[:self.config.MAX_QLEN - 1], np.array([4999])])

    def mapper2(self, tokenized):
        dis = tf.py_function(
            self.py_discourse_marker_mapper,
            inp=[tokenized['context']], Tout=tf.uint8)
        dis.set_shape([self.config.MAX_CONTEXT_LEN, ])
        cidx0 = self.word2idx0.lookup(tokenized['context'])
        cidx = tf.py_function(
            self.py_pad_context,
            inp=[cidx0], Tout=tf.int32
        )
        cidx.set_shape([self.config.MAX_CONTEXT_LEN, ])
        qidx0 = self.qword2idx0.lookup(tokenized['question'])
        qidx = tf.py_function(
            self.py_pad_question,
            inp=[qidx0], Tout=tf.int32
        )
        qidx.set_shape([self.config.MAX_QLEN + 1, ])
        feature = {
            "cidx": cidx,
            "cdis": dis,
            "qidx": qidx,
        }
        return feature

    @staticmethod
    def serialize(cidx, cdis, qidx):
        cidx = tf.io.serialize_tensor(cidx)
        qidx = tf.io.serialize_tensor(qidx)
        cdis = tf.io.serialize_tensor(cdis)
        feature = {
            "cidx": QuestionContextPairs._bytes_feature(cidx),
            "cdis": QuestionContextPairs._bytes_feature(cdis),
            "qidx": QuestionContextPairs._bytes_feature(qidx),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def mapper3(self, features):
        tf_string = tf.py_function(
            QuestionContextPairs.serialize,
            [features['cidx'], features['cdis'], features['qidx']],
            tf.string)    # the return type is `tf.string`.
        return tf.reshape(tf_string, ())  # The result is a scalar

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} should be a directory!")
        print("******** Saving Validation set ********")
        fname = os.path.join(folder, "QCPair.val.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(self.val.map(self.mapper3, num_parallel_calls=-1))

        print("******** Saving Training set ********")
        fname = os.path.join(folder, "QCPair.train.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(self.train.map(self.mapper3, num_parallel_calls=-1))
        print("******** Finished saving dataset ********")

    @staticmethod
    def parse_ex(example_proto):
        feature_description = {
            'cidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'cdis': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qidx': tf.io.FixedLenFeature([], tf.string, default_value='')
        }
        example = tf.io.parse_single_example(
            example_proto, feature_description)
        cidx = tf.io.parse_tensor(example['cidx'], out_type=tf.int32)
        cidx.set_shape([CONFIG.MAX_CONTEXT_LEN, ])
        qidx = tf.io.parse_tensor(example['qidx'], out_type=tf.int32)
        cdis = tf.io.parse_tensor(example['cdis'], out_type=tf.uint8)
        cdis.set_shape([CONFIG.MAX_CONTEXT_LEN, ])
        randin = tf.random.normal((CONFIG.LATENT_DIM,))
        qidx.set_shape([CONFIG.MAX_QLEN+1, ])
        return ((cidx, cdis, randin, qidx[:-1]), qidx[1:])

    # @staticmethod
    # def flatten_all(X, qidx):
    #     cidx, cdis, randin = X
    #     X1 = []
    #     X2 = []
    #     X3 = []
    #     X4 = []
    #     y = []
    #     for i in range(1, qidx.shape[0]):
    #         X1.append(cidx)
    #         X2.append(cdis)
    #         X3.append(randin)
    #         X4.append(qidx[:i])
    #         y.append(qidx[i])
    #     X1 = tf.data.Dataset.from_tensor_slices(X1)
    #     X2 = tf.data.Dataset.from_tensor_slices(X2)
    #     X3 = tf.data.Dataset.from_tensor_slices(X3)
    #     x4 = tf.data.Dataset.from_tensors(X4[0])
    #     for x4i in X4[1:]:
    #         x4 = x4.concatenate(tf.data.Dataset.from_tensors(x4i))
    #     y = tf.data.Dataset.from_tensor_slices(y)
    #     X = tf.data.Dataset.zip((X1, X2))
    #     X = tf.data.Dataset.zip((X, X3))
    #     X = tf.data.Dataset.zip((X, x4))
    #     return tf.data.Dataset.zip((X, y))

    # @staticmethod
    # def reshape_mapper(X, y):
    #     X123, X4 = X
    #     X12, X3 = X123
    #     X1, X2 = X12
    #     return ((X1, X2, X3, X4), y)

    @classmethod
    def load(cls, folder):
        if not os.path.exists(folder) or not os.path.isdir(folder):
            raise ValueError(f"{folder} should be a directory!")
        TRAIN = os.path.join(folder, "QCPair.train.tfrecord")
        VAL = os.path.join(folder, "QCPair.val.tfrecord")
        if not os.path.exists(TRAIN) or not os.path.exists(VAL):
            raise ValueError(f"Dataset not present inside {folder}!")
        train = tf.data.TFRecordDataset([TRAIN], compression_type='ZLIB')
        val = tf.data.TFRecordDataset([VAL], compression_type='ZLIB')
        train = train.map(cls.parse_ex, num_parallel_calls=-1)
        # train = train.flat_map(cls.flatten_all)\
        #     .map(cls.reshape_mapper, num_parallel_calls=-1)
        val = val.map(cls.parse_ex, num_parallel_calls=-1)
        # val = val.flat_map(cls.flatten_all)\
        #     .map(cls.reshape_mapper, num_parallel_calls=-1)
        inst = cls(None, create=False)
        inst.train = train
        inst.val = val
        return inst
