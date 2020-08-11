import spacy
import logging
from copynet_tf import Vocab
import numpy as np
import tensorflow as tf
import gc
import os

from ..config import cfg
from ..utils import DownloadManager, SQuADReader
from ..features import NERTagger, PosTagger, GloVeReader, FastTextReader


class SQuAD_CA_PreQC:
    def __init__(self, prepare=False):
        self.logger = logging.getLogger(__name__)
        if prepare:
            self._prepare()
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
            'cis': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'cit': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'ans': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qpre': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qit': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qis': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'ner': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'pos': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        example = tf.io.parse_single_example(
            example_proto, feature_description)
        cis = tf.io.parse_tensor(example['cis'], out_type=tf.int32)
        cis.set_shape([cfg.CSEQ_LEN, ])
        cit = tf.io.parse_tensor(example['cit'], out_type=tf.int32)
        cit.set_shape([cfg.CSEQ_LEN, ])
        qit = tf.io.parse_tensor(example['qit'], out_type=tf.int32)
        qit.set_shape([cfg.QSEQ_LEN, ])
        qis = tf.io.parse_tensor(example['qis'], out_type=tf.int32)
        qis.set_shape([cfg.QSEQ_LEN, ])
        ans = tf.io.parse_tensor(example['ans'], out_type=tf.uint8)
        ans.set_shape([cfg.CSEQ_LEN, ])
        ner = tf.io.parse_tensor(example['ner'], out_type=tf.uint8)
        ner.set_shape([cfg.CSEQ_LEN, ])
        pos = tf.io.parse_tensor(example['pos'], out_type=tf.uint8)
        pos.set_shape([cfg.CSEQ_LEN, ])
        qpre = tf.io.parse_tensor(example['qpre'], out_type=tf.int32)
        qpre.set_shape([cfg.QSEQ_LEN, ])
        return ((cis, cit, ans, ner, pos, qpre), (qit, qis))

    def substrSearch(self, ans, context):
        i = 0
        j = 0
        s = -1
        while i < len(context) and j < len(ans):
            if context[i].text == ans[j].text:
                if s == -1:
                    s = i
                i += 1
                j += 1
            else:
                i += 1
                j = 0
                s = -1

        return s, j

    def _prepare(self):
        self.logger.info("****Preparing dataset****")
        self._download()
        self.logger.info("****Parse dataset****")
        raw_train = self._read(cfg.RAW_TRAIN_SAVE)
        raw_test = self._read(cfg.RAW_DEV_SAVE)
        train_context, train_question, train_ans = self._unzip(raw_train)
        test_context, test_question, test_ans = self._unzip(raw_test)

        if cfg.EMBS_TYPE == 'glove':
            embedding_reader = GloVeReader()
        elif cfg.EMBS_TYPE == 'fasttext':
            embedding_reader = FastTextReader()
        else:
            raise ValueError(f"Unsupported embeddings type {cfg.EMBS_TYPE}")
        pretrained_vectors = embedding_reader.read(cfg.EMBS_FILE)

        nlp = spacy.load("en_core_web_sm")
        ner = NERTagger(cfg.NER_TAGS_FILE, cfg.CSEQ_LEN)
        pos = PosTagger(cfg.POS_TAGS_FILE, cfg.CSEQ_LEN)
        vocab = Vocab(
            embedding_reader.START,
            embedding_reader.END,
            embedding_reader.PAD,
            embedding_reader.UNK,
            cfg.CSEQ_LEN,
            cfg.QSEQ_LEN
        )

        self.logger.info("****Prepare train set****")
        train_context = nlp.pipe(
            train_context, batch_size=256, n_process=cfg.MAX_PARALLELISM)
        train_question = nlp.pipe(
            train_question, batch_size=256, n_process=cfg.MAX_PARALLELISM)
        train_ans = nlp.pipe(
            train_ans, batch_size=256, n_process=cfg.MAX_PARALLELISM)

        training_context = []
        training_question = []
        training_preq = []
        training_fullq = []
        training_ans = []
        noun = 'NOUN'
        propn = 'PROPN'
        verb = 'VERB'
        # BIO tagging for answer
        for context, ques, ans in zip(
                train_context, train_question, train_ans):
            ans_start, al = self.substrSearch(ans, context)
            ans_start += 1
            if len(ques) >= 20 or ans_start == -1 or ans_start + al >= 250:
                continue
            qtags = [token.pos_ for token in ques]
            split = len(qtags)//2
            if verb in qtags:
                split = qtags.index(verb) + 1
            else:
                posnoun = 21
                pospropn = 21
                try:
                    posnoun = qtags.index(noun) + 1
                except:  # noqa
                    pass
                try:
                    pospropn = qtags.index(propn) + 1
                except:  # noqa
                    pass
                idx = min(posnoun, pospropn)
                if idx != 21:
                    split = idx
            split = min(split, 8)
            if len(ques[split:]) > 8:
                continue
            training_context.append(context)
            training_question.append(ques[split:])
            training_preq.append(ques[:split])
            training_fullq.append(ques)
            ans = np.zeros(cfg.CSEQ_LEN, dtype=np.uint8)
            ans[ans_start] = 1
            ans[ans_start+1:ans_start+al] = 2
            training_ans.append(ans)

        vocab.fit(
            training_context,
            training_fullq,
            pretrained_vectors,
            0, cfg.MIN_QVOCAB_FREQ
        )
        train_cis = vocab.transform(training_context, "source")
        train_cit = vocab.transform(training_context, "target", cfg.CSEQ_LEN)
        train_ner = ner.transform(training_context)
        train_pos = pos.transform(training_context)
        train_qit = vocab.transform(training_question, "target")
        train_qis = vocab.transform(training_question, "source", cfg.QSEQ_LEN)
        train_qpre = vocab.transform(training_preq, "target")

        cseq = cfg.CSEQ_LEN
        qseq = cfg.QSEQ_LEN

        def gen():
            for cis, cit, ner, pos, qit, qis, ans, preq in zip(
                    train_cis, train_cit, train_ner, train_pos,
                    train_qit, train_qis, training_ans, train_qpre):
                yield (cis, cit, ner, pos, qit, qis, ans, preq)

        train_dataset = tf.data.Dataset.from_generator(
            gen,
            (
                tf.int32, tf.int32, tf.uint8, tf.uint8,
                tf.int32, tf.int32, tf.uint8, tf.int32),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([qseq]),
                tf.TensorShape([cseq]), tf.TensorShape([qseq]))
        )

        self.logger.info("****Prepare test set****")
        test_context = nlp.pipe(
            test_context, batch_size=256, n_process=cfg.MAX_PARALLELISM)
        test_question = nlp.pipe(
            test_question, batch_size=256, n_process=cfg.MAX_PARALLELISM)
        test_ans = nlp.pipe(
            test_ans, batch_size=256, n_process=cfg.MAX_PARALLELISM)

        testing_context = []
        testing_question = []
        testing_preq = []
        testing_fullq = []
        testing_ans = []
        noun = 'NOUN'
        propn = 'PROPN'
        verb = 'VERB'
        # BIO tagging for answer
        for context, ques, ans in zip(
                test_context, test_question, test_ans):
            ans_start, al = self.substrSearch(ans, context)
            ans_start += 1
            if len(ques) >= 20 or ans_start == -1 or ans_start + al >= 250:
                continue
            qtags = [token.pos_ for token in ques]
            split = len(qtags)//2
            if verb in qtags:
                split = qtags.index(verb) + 1
            else:
                posnoun = 21
                pospropn = 21
                try:
                    posnoun = qtags.index(noun) + 1
                except:  # noqa
                    pass
                try:
                    pospropn = qtags.index(propn) + 1
                except:  # noqa
                    pass
                idx = min(posnoun, pospropn)
                if idx != 21:
                    split = idx
            split = min(split, 8)
            if len(ques[split:]) > 8:
                continue
            testing_context.append(context)
            testing_question.append(ques[split:])
            testing_preq.append(ques[:split])
            testing_fullq.append(ques)
            ans = np.zeros(cfg.CSEQ_LEN, dtype=np.uint8)
            ans[ans_start] = 1
            ans[ans_start+1:ans_start+al] = 2
            testing_ans.append(ans)

        test_cis = vocab.transform(testing_context, "source")
        test_cit = vocab.transform(testing_context, "target", cfg.CSEQ_LEN)
        test_ner = ner.transform(testing_context)
        test_pos = pos.transform(testing_context)
        test_qit = vocab.transform(testing_question, "target")
        test_qis = vocab.transform(testing_question, "source", cfg.QSEQ_LEN)
        test_qpre = vocab.transform(testing_preq, "target")

        cseq = cfg.CSEQ_LEN
        qseq = cfg.QSEQ_LEN

        def gen():
            for cis, cit, ner, pos, qit, qis, ans, preq in zip(
                    test_cis, test_cit, test_ner, test_pos,
                    test_qit, test_qis, testing_ans, test_qpre):
                yield (cis, cit, ner, pos, qit, qis, ans, preq)

        test_dataset = tf.data.Dataset.from_generator(
            gen,
            (
                tf.int32, tf.int32, tf.uint8, tf.uint8,
                tf.int32, tf.int32, tf.uint8, tf.int32),
            (
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([cseq]), tf.TensorShape([cseq]),
                tf.TensorShape([qseq]), tf.TensorShape([qseq]),
                tf.TensorShape([cseq]), tf.TensorShape([qseq]))
        )

        train_dataset = train_dataset.map(
            self.make_example, num_parallel_calls=-1)
        test_dataset = test_dataset.map(
            self.make_example, num_parallel_calls=-1)

        self.logger.debug(f"****Saving dataset****")
        self.save(train_dataset, test_dataset)
        vocab.save(cfg.VOCAB_SAVE)
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

    def make_example(self, cis, cit, ner, pos, qit, qis, ans, qpre):
        serialized = tf.py_function(
            self.serialize,
            [cis, cit, ner, pos, qit, qis, ans, qpre],
            tf.string
        )
        return tf.reshape(serialized, ())

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize(self, cis, cit, ner, pos, qit, qis, ans, qpre):
        cis = tf.io.serialize_tensor(cis)
        cit = tf.io.serialize_tensor(cit)
        ans = tf.io.serialize_tensor(ans)
        qit = tf.io.serialize_tensor(qit)
        qis = tf.io.serialize_tensor(qis)
        ner = tf.io.serialize_tensor(ner)
        pos = tf.io.serialize_tensor(pos)
        qpre = tf.io.serialize_tensor(qpre)
        feature = {
            "cis": self._bytes_feature(cis),
            "cit": self._bytes_feature(cit),
            "ans": self._bytes_feature(ans),
            "qit": self._bytes_feature(qit),
            "qis": self._bytes_feature(qis),
            "ner": self._bytes_feature(ner),
            "pos": self._bytes_feature(pos),
            "qpre": self._bytes_feature(qpre)
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def _download(self):
        self.logger.info("****Maybe download****")
        manager = DownloadManager()
        manager.download(cfg.RAW_TRAIN_URL, cfg.RAW_TRAIN_SAVE)
        manager.download(cfg.RAW_DEV_URL, cfg.RAW_DEV_SAVE)

    def _read(self, filename):
        reader = SQuADReader()
        parsed = reader.parse(filename)
        filtered = reader.filter_unique_ca_pairs(parsed)
        return reader.flatten_parsed(filtered)

    def _unzip(self, raw):
        context = map(lambda x: x["context"], raw)
        question = map(lambda x: x["question"], raw)
        answer = map(lambda x: x["answer"], raw)
        return context, question, answer
