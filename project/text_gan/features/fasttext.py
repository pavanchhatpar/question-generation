from .embedding import Embedding
from ..config import cfg

import numpy as np
from tqdm import tqdm
import logging
import pickle
import os


class FastText(Embedding):
    UNK = 'UNKNOWN'
    PAD = 'PAD'
    START = 'S'
    END = 'EOS'

    def __init__(self, embeddings_file, sequence_len, loaded_embeddings=None):
        super(FastText, self).__init__()
        self.logger = logging.getLogger(__name__)
        if loaded_embeddings is not None:
            self.data = loaded_embeddings
            self.n = len(self.data)
            self.d = 300
        else:
            if os.path.exists(cfg.EMBS_CACHE):
                with open(cfg.EMBS_CACHE, "rb") as f:
                    self.data = pickle.load(f)
                self.n = len(self.data)
                self.d = 300
            else:
                with open(embeddings_file, 'r') as fin:
                    self.n, self.d = map(int, fin.readline().split())
                    self.data = {}
                    for line in tqdm(fin, desc='Loading vectors'):
                        tokens = line.rstrip().split(' ')
                        self.data[tokens[0].strip()] = np.array(
                            tokens[1:], dtype=np.float32)
                self._spl_token_report()
                self.cache()
        self.seq_len = sequence_len

    @classmethod
    def load(
            cls, embeddings_file, sequence_len, vocab_file,
            loaded_embeddings=None):
        inst = cls(embeddings_file, sequence_len, loaded_embeddings)
        with open(vocab_file, 'rb') as f:
            inst.vocab = pickle.load(f)
        inst.inverse = {}
        for k, v in inst.vocab.items():
            inst.inverse[v] = k
        return inst
