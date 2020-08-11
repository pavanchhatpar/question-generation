import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

model_name = 'canp_qc'

# Start defining default config
__C.MODEL = 'canp-qc'
__C.CONFIG_NAME = 'DEFAULT'
__C.MAX_ARRAY_LOG = 10
__C.SAVE_LOC = f'./data/{model_name}/features'
__C.MODEL_SAVE = f'./data/{model_name}/model'

__C.RAW_TRAIN_URL = ('https://raw.githubusercontent.com/rajpurkar/SQuAD-explor'
                     + 'er/master/dataset/train-v1.1.json')
__C.RAW_DEV_URL = ('https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer'
                   + '/master/dataset/dev-v1.1.json')

__C.RAW_TRAIN_SAVE = './data/squad1/train.json'
__C.RAW_DEV_SAVE = './data/squad1/dev.json'
__C.MAX_PARALLELISM = 12

__C.CSEQ_LEN = 250
__C.QSEQ_LEN = 20
__C.BEAM_WIDTH = 2
__C.QPRED_LEN = 19
__C.MIN_QVOCAB_FREQ = 5
__C.CKPT_COUNT = 10

__C.EMBS_TYPE = 'glove'
__C.EMBS_FILE = './data/features/glove/glove.840B.300d.txt'
__C.VOCAB_SAVE = f'./data/{model_name}/features/vocab'
__C.NER_TAGS_FILE = './data/features/ner/vocab.txt'
__C.POS_TAGS_FILE = './data/features/postags/vocab.txt'

__C.UNK_ID = 1
__C.PAD_ID = 0
__C.START_ID = 2
__C.END_ID = 3

__C.BATCH_SIZE = 128
__C.LATENT_DIM = 300
__C.HIDDEN_DIM = 300
__C.LR = 1e-3
__C.CLIP_NORM = 1
__C.DROPOUT = 0.3
__C.EPOCHS = 100
__C.TRAIN_SIZE = 10000
__C.VAL_SIZE = 1000
__C.STEPS_PER_EPOCH = -1
# End defining default config


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:  # noqa
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
