from typing import Dict
import numpy as np
from tqdm import tqdm

from .embedding_reader import EmbeddingReader


class FastTextReader(EmbeddingReader):
    UNK = 'UNKNOWN'
    PAD = 'PAD'
    START = 'S'
    END = 'EOS'

    def read(self,
             filename: str) -> Dict[str, np.ndarray]:
        data = {}
        with open(filename, 'r') as fin:
            n, d = map(int, fin.readline().split())
            for line in tqdm(fin, desc='Loading vectors'):
                tokens = line.rstrip().split(' ')
                data[tokens[0].strip()] = np.array(
                    tokens[1:], dtype=np.float32)
        return data
