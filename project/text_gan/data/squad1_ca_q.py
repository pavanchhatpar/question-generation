from ..config import cfg
from ..features import GloVe, FastText, NERTagger, PosTagger
from ..utils import MapReduce

import en_core_web_sm
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import logging
import os
import gc


class Squad1_CA_Q:
    def __init__(self, prepare=False):
        self.logger = logging.getLogger(__name__)
        if prepare:
            self.preprocess()
        if not os.path.exists(cfg.SAVE_LOC) or not os.path.isdir(cfg.SAVE_LOC):
            raise ValueError(f"{cfg.SAVE_LOC} should be a directory!")
        TRAIN = os.path.join(cfg.SAVE_LOC, "train.tfrecord")
        TEST = os.path.join(cfg.SAVE_LOC, "test.tfrecord")
        if not os.path.exists(TRAIN) or not os.path.exists(TEST):
            raise ValueError(f"Dataset not present inside {cfg.SAVE_LOC}!")
        train = tf.data.TFRecordDataset([TRAIN], compression_type='ZLIB')
        test = tf.data.TFRecordDataset([TEST], compression_type='ZLIB')
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train = train.map(self.parse_ex, num_parallel_calls=AUTOTUNE)
        self.test = test.map(self.parse_ex, num_parallel_calls=AUTOTUNE)

    def __getstate__(self):
        dic = self.__dict__.copy()
        del dic['logger']
        return dic

    def __setstate__(self, dic):
        self.__dict__.update(dic)
        self.logger = logging.getLogger(__name__)

    def parse_ex(self, example_proto):
        feature_description = {
            'cidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'aidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qidx': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'ner': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'pos': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        example = tf.io.parse_single_example(
            example_proto, feature_description)
        cidx = tf.io.parse_tensor(example['cidx'], out_type=tf.int32)
        cidx.set_shape([cfg.CSEQ_LEN, ])
        qidx = tf.io.parse_tensor(example['qidx'], out_type=tf.int32)
        qidx.set_shape([cfg.QSEQ_LEN, ])
        aidx = tf.io.parse_tensor(example['aidx'], out_type=tf.uint8)
        aidx.set_shape([cfg.CSEQ_LEN, ])
        ner = tf.io.parse_tensor(example['ner'], out_type=tf.uint8)
        ner.set_shape([cfg.CSEQ_LEN, ])
        pos = tf.io.parse_tensor(example['pos'], out_type=tf.uint8)
        pos.set_shape([cfg.CSEQ_LEN, ])
        return ((cidx, aidx, ner, pos), qidx)

    def tokenize_context(self, x):
        return self.nlp(x['context'].decode('utf-8'))

    def tokenize_question(self, x):
        return self.nlp(x['question'].decode('utf-8'))

    def tokenize_answer(self, x):
        return self.nlp(x['answers']['text'][0].decode('utf-8'))

    def tag_answer(self, inp):
        cidx, aidx = inp
        aidx = np.array(aidx, dtype=np.uint8)
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

    def preprocess(self):
        self.logger.info("****Preparing dataset****")
        if cfg.EMBS_TYPE == 'glove':
            cembs = GloVe(cfg.EMBS_FILE, cfg.CSEQ_LEN)
            qembs = GloVe(cfg.EMBS_FILE, cfg.QSEQ_LEN, cembs.data)
        elif cfg.EMBS_TYPE == 'fasttext':
            cembs = FastText(cfg.EMBS_FILE, cfg.CSEQ_LEN)
            qembs = FastText(cfg.EMBS_FILE, cfg.QSEQ_LEN, cembs.data)
        else:
            raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
        ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
        pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)
        self.nlp = en_core_web_sm.load()

        train = tfds.load("squad", data_dir="/tf/data/tf_data", split='train')
        test = tfds.load(
            "squad", data_dir="/tf/data/tf_data", split='validation')

        mr = MapReduce()

        train_context = train.as_numpy_iterator()
        test_context = test.as_numpy_iterator()

        train_context = mr.process(self.tokenize_context, train_context)
        test_context = mr.process(self.tokenize_context, test_context)
        self.logger.info("****Tokenized context****")
        cembs.fit(train_context, min_freq=None)
        train_cembs = cembs.transform(train_context)
        test_cembs = cembs.transform(test_context)
        train_ner = ner.transform(train_context)
        test_ner = ner.transform(test_context)
        train_pos = pos.transform(train_context)
        test_pos = pos.transform(test_context)
        self.logger.info("****Prepared context****")
        self.logger.debug(f"Memory freed: {gc.collect()}")

        train_question = train.as_numpy_iterator()
        test_question = test.as_numpy_iterator()

        train_question = mr.process(self.tokenize_question, train_question)
        test_question = mr.process(self.tokenize_question, test_question)
        self.logger.info("****Tokenized question****")
        qembs.fit(train_question, min_freq=None)
        train_qembs = qembs.transform(train_question)
        test_qembs = qembs.transform(test_question)
        self.logger.info("****Prepared question****")
        self.logger.debug(f"Memory freed: {gc.collect()}")

        train_answer = train.as_numpy_iterator()
        test_answer = test.as_numpy_iterator()

        train_answer = mr.process(self.tokenize_answer, train_answer)
        test_answer = mr.process(self.tokenize_answer, test_answer)
        self.logger.info("****Tokenized answer****")

        train_aembs = cembs.transform(train_answer)
        test_aembs = cembs.transform(test_answer)

        train_aembs = mr.process(
            self.tag_answer, zip(train_cembs, train_aembs))
        train_aembs = np.array(train_aembs, dtype=np.uint8)
        test_aembs = mr.process(self.tag_answer, zip(test_cembs, test_aembs))
        test_aembs = np.array(test_aembs, dtype=np.uint8)
        self.logger.info("****Prepared answer****")
        self.logger.debug(f"Memory freed: {gc.collect()}")

        cseq = cfg.CSEQ_LEN
        qseq = cfg.QSEQ_LEN

        def gen():
            for cidx, aidx, qidx, ner, pos in zip(
                    train_cembs, train_aembs,
                    train_qembs, train_ner, train_pos):
                yield (cidx, aidx, qidx, ner, pos)

        train_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.uint8, tf.int32, tf.uint8, tf.uint8),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([cseq]),
                tf.TensorShape([cseq]))
        )

        def gen():
            for cidx, aidx, qidx, ner, pos in zip(
                    test_cembs, test_aembs,
                    test_qembs, test_ner, test_pos):
                yield (cidx, aidx, qidx, ner, pos)

        test_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.uint8, tf.int32, tf.uint8, tf.uint8),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([cseq]),
                tf.TensorShape([cseq]))
        )

        train_dataset = train_dataset.map(
            self.make_example, num_parallel_calls=-1)
        test_dataset = test_dataset.map(
            self.make_example, num_parallel_calls=-1)

        cembs.save(cfg.EMBS_CVOCAB)
        qembs.save(cfg.EMBS_QVOCAB)
        self.save(train_dataset, test_dataset)
        self.logger.debug(f"Memory freed: {gc.collect()}")

    def save(self, train, test):
        if not os.path.exists(cfg.SAVE_LOC):
            os.makedirs(cfg.SAVE_LOC)
        if not os.path.isdir(cfg.SAVE_LOC):
            raise ValueError(f"{cfg.SAVE_LOC} should be a directory!")
        self.logger.info("******** Saving Test set ********")
        fname = os.path.join(cfg.SAVE_LOC, "test.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(test)

        self.logger.info("******** Saving Training set ********")
        fname = os.path.join(cfg.SAVE_LOC, "train.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(train)
        self.logger.info("******** Finished saving dataset ********")

    def make_example(self, cidx, aidx, qidx, ner, pos):
        serialized = tf.py_function(
            self.serialize,
            [cidx, aidx, qidx, ner, pos],
            tf.string
        )
        return tf.reshape(serialized, ())

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize(self, cidx, aidx, qidx, ner, pos):
        cidx = tf.io.serialize_tensor(cidx)
        aidx = tf.io.serialize_tensor(aidx)
        qidx = tf.io.serialize_tensor(qidx)
        ner = tf.io.serialize_tensor(ner)
        pos = tf.io.serialize_tensor(pos)
        feature = {
            "cidx": self._bytes_feature(cidx),
            "aidx": self._bytes_feature(aidx),
            "qidx": self._bytes_feature(qidx),
            "ner": self._bytes_feature(ner),
            "pos": self._bytes_feature(pos),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
