from abc import ABC
from scipy.spatial.distance import cosine
from collections import defaultdict
import pickle
import numpy as np

from ..config import cfg


class Embedding(ABC):
    UNK_ID = cfg.UNK_ID
    PAD_ID = cfg.PAD_ID
    START_ID = cfg.START_ID
    END_ID = cfg.END_ID

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data[key]

    def __missing__(self, key):
        return self.data[self.UNK]

    def _spl_token_report(self):
        apt = 0.7
        try:
            unk = self[self.UNK]
            pad = self[self.PAD]
            start = self[self.START]
            end = self[self.END]
            dis1 = cosine(unk, pad)
            dis2 = cosine(unk, start)
            dis3 = cosine(unk, end)
            dis4 = cosine(pad, start)
            dis5 = cosine(pad, end)
            dis6 = cosine(start, end)
            self.logger.debug(f"DISTANCE({self.UNK}, {self.PAD}) = {dis1}")
            self.logger.debug(f"DISTANCE({self.UNK}, {self.START}) = {dis2}")
            self.logger.debug(f"DISTANCE({self.UNK}, {self.END}) = {dis3}")
            self.logger.debug(f"DISTANCE({self.PAD}, {self.START}) = {dis4}")
            self.logger.debug(f"DISTANCE({self.PAD}, {self.END}) = {dis5}")
            self.logger.debug(f"DISTANCE({self.START}, {self.END}) = {dis6}")
            if dis1 < apt:
                self.logger.warn(
                    f"DISTANCE({self.UNK}, {self.PAD}) = {dis1}")
            if dis2 < apt:
                self.logger.warn(
                    f"DISTANCE({self.UNK}, {self.START}) = {dis2}")
            if dis3 < apt:
                self.logger.warn(
                    f"DISTANCE({self.UNK}, {self.END}) = {dis3}")
            if dis4 < apt:
                self.logger.warn(
                    f"DISTANCE({self.PAD}, {self.START}) = {dis4}")
            if dis5 < apt:
                self.logger.warn(
                    f"DISTANCE({self.PAD}, {self.END}) = {dis5}")
            if dis6 < apt:
                self.logger.warn(
                    f"DISTANCE({self.START}, {self.END}) = {dis6}")
        except AttributeError as e:
            err = "Define UNK, PAD, START, END special tokens on your class"
            self.logger.error(err)
            raise(e)
        except KeyError as e:
            err = ("Values defined for UNK, PAD, START, END"
                   + " should have embeddings")
            self.logger.error(err)
            raise(e)

    def fit(self, ntokens, min_freq=None):
        vocab = defaultdict(lambda: 0)
        for doc in ntokens:
            for token in doc:
                vocab[token.text] += 1

        if min_freq is not None:
            vocab = dict(filter(lambda x: x[1] > min_freq, vocab.items()))
        self.vocab = {}
        self.vocab[self.UNK] = self.UNK_ID
        self.vocab[self.PAD] = self.PAD_ID
        self.vocab[self.START] = self.START_ID
        self.vocab[self.END] = self.END_ID
        i = 4
        for token, freq in vocab.items():
            if token in self.vocab:
                continue
            self.vocab[token] = i
            i += 1
        self.inverse = {}
        for k, v in vocab.items():
            self.inverse[v] = k

    def transform(self, ntokens, pad=True, end=True):
        nids = []
        for tokens in ntokens:
            if end:
                ids = [self.vocab.get(
                    token.text,
                    self.UNK_ID) for token in tokens[:self.seq_len-1]]
                ids.append(self.END_ID)
            else:
                ids = [self.vocab.get(
                    token.text,
                    self.UNK_ID) for token in tokens[:self.seq_len]]
            if pad:
                while len(ids) < self.seq_len:
                    ids.append(self.PAD_ID)
            nids.append(ids)
        if pad:
            return np.array(nids, dtype=np.int32)
        else:
            return nids

    def inverse_transform(self, nids):
        ntokens = []
        for ids in nids:
            tokens = [self.inverse.get(id, self.UNK) for id in ids]
            ntokens.append(tokens)
        return ntokens

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f, protocol=4)

    def get_matrix(self):
        shape = (len(self.vocab), self.d)
        matrix = np.zeros(shape, dtype=np.float32)
        for token, idx in self.vocab.items():
            matrix[idx] = self.data.get(token, self.data[self.UNK])
        return matrix
