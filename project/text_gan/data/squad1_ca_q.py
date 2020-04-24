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
            'ans': tf.io.FixedLenFeature([], tf.string, default_value=''),
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
        ans = tf.io.parse_tensor(example['ans'], out_type=tf.uint8)
        ans.set_shape([cfg.CSEQ_LEN, ])
        ner = tf.io.parse_tensor(example['ner'], out_type=tf.uint8)
        ner.set_shape([cfg.CSEQ_LEN, ])
        pos = tf.io.parse_tensor(example['pos'], out_type=tf.uint8)
        pos.set_shape([cfg.CSEQ_LEN, ])
        return ((cidx, ans, ner, pos), qidx)

    def tokenize_example(self, x):
        context, question, ans = list(self.nlp.pipe([
            x['context'].decode('utf-8'),
            x['question'].decode('utf-8'),
            x['answers']['text'][0].decode('utf-8')
        ]))
        del x
        return (context, question, ans)

    def substrSearch(self, ans, context):
        i = 0
        j = 0
        s = -1
        while i < len(context) and j < len(ans):
            if context[i].text == ans[j].text:
                s = i
                i += 1
                j += 1
            else:
                i += 1
                j = 0
                s = -1
        return s, j

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

        self.logger.info("****Preparing training split****")
        train_iter = train.as_numpy_iterator()
        train_tokenized = mr.process(self.tokenize_example, train_iter)
        self.logger.info("****Tokenized training split****")

        train_context = []
        train_question = []
        train_ans = []
        for context, ques, ans in train_tokenized:
            ans_start, al = self.substrSearch(ans, context)
            if len(ques) >= 20 or ans_start == -1 or ans_start + al >= 250:
                continue
            train_context.append(context)
            train_question.append(ques)
            ans = np.zeros(cfg.CSEQ_LEN, dtype=np.uint8)
            ans[ans_start:ans_start+al] = 1
            train_ans.append(ans)
        self.logger.info("****Filtered training split****")

        cembs.fit(train_context)
        qembs.fit(train_question, min_freq=2)
        cembs.save(cfg.EMBS_CVOCAB)
        qembs.save(cfg.EMBS_QVOCAB)
        train_cidx = cembs.transform(train_context)
        train_ner = ner.transform(train_context)
        train_pos = pos.transform(train_context)
        train_qidx = qembs.transform(train_question)

        cseq = cfg.CSEQ_LEN
        qseq = cfg.QSEQ_LEN

        def gen():
            for cidx, ner, pos, qidx, ans in zip(
                    train_cidx, train_ner, train_pos,
                    train_qidx, train_ans):
                yield (cidx, ans, qidx, ner, pos)

        train_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.uint8, tf.int32, tf.uint8, tf.uint8),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([cseq]), tf.TensorShape([cseq]))
        )

        self.logger.info("****Preparing test split****")
        test_iter = test.as_numpy_iterator()
        test_tokenized = mr.process(self.tokenize_example, test_iter)
        self.logger.info("****Tokenized test split****")

        test_context = []
        test_question = []
        test_ans = []
        for context, ques, ans in test_tokenized:
            ans_start, al = self.substrSearch(ans, context)
            if len(ques) >= 20 or ans_start == -1 or ans_start + al >= 250:
                continue
            test_context.append(context)
            test_question.append(ques)
            ans = np.zeros(cfg.CSEQ_LEN, dtype=np.uint8)
            ans[ans_start:ans_start+al] = 1
            test_ans.append(ans)
        self.logger.info("****Filtered test split****")

        test_cidx = cembs.transform(test_context)
        test_ner = ner.transform(test_context)
        test_pos = pos.transform(test_context)
        test_qidx = qembs.transform(test_question)

        cseq = cfg.CSEQ_LEN
        qseq = cfg.QSEQ_LEN

        def gen():
            for cidx, ner, pos, qidx, ans in zip(
                    test_cidx, test_ner, test_pos,
                    test_qidx, test_ans):
                yield (cidx, ans, qidx, ner, pos)

        test_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.uint8, tf.int32, tf.uint8, tf.uint8),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([cseq]), tf.TensorShape([cseq]))
        )

        train_dataset = train_dataset.map(
            self.make_example, num_parallel_calls=-1)
        test_dataset = test_dataset.map(
            self.make_example, num_parallel_calls=-1)

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

    def make_example(self, cidx, ans, qidx, ner, pos):
        serialized = tf.py_function(
            self.serialize,
            [cidx, ans, qidx, ner, pos],
            tf.string
        )
        return tf.reshape(serialized, ())

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize(self, cidx, ans, qidx, ner, pos):
        cidx = tf.io.serialize_tensor(cidx)
        ans = tf.io.serialize_tensor(ans)
        qidx = tf.io.serialize_tensor(qidx)
        ner = tf.io.serialize_tensor(ner)
        pos = tf.io.serialize_tensor(pos)
        feature = {
            "cidx": self._bytes_feature(cidx),
            "ans": self._bytes_feature(ans),
            "qidx": self._bytes_feature(qidx),
            "ner": self._bytes_feature(ner),
            "pos": self._bytes_feature(pos),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
