from ..config import cfg

import numpy as np
import logging


class NERTagger:
    def __init__(self, tags_file, sequence_len):
        self.logger = logging.getLogger(__name__)
        with open(tags_file, 'r') as f:
            self.tags2idx = {}
            i = 0
            for line in f:
                tag = line.split('\t')[0]
                self.tags2idx[tag] = i
                i += 1
            self.tags2idx[''] = i
        self.seq_len = sequence_len

    def transform(self, ntokens):
        ntags = []
        logged = 0
        for tokens in ntokens:
            tokens = tokens[:self.seq_len]
            tags = [token.ent_type_ for token in tokens]
            if logged < cfg.MAX_ARRAY_LOG and np.random.uniform() < 0.1:
                debug = f"{[t.text for t in tokens]}\n\n{tags}\n\n\n"
                self.logger.debug(debug)
                logged += 1
            while len(tags) < self.seq_len:
                tags.append('')
            tags[-1] = ''
            tags = [self.tags2idx[tag] for tag in tags]
            ntags.append(tags)
        return np.array(ntags, dtype=np.uint8)
