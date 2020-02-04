from tensorflow_datasets.text import Squad
import tensorflow_datasets as tfds
import tensorflow as tf

_CITATION = """\
@article{rajpurkar2018know,
       author = {{Rajpurkar}, Pranav and {Jia}, Robin and {Liang}, Percy},
        title = "{Know What You Don't Know: Unanswerable Questions for SQuAD}",
      journal = {arXiv e-prints},
         year = 2018,
          eid = {arXiv:1806.03822},
        pages = {arXiv:1806.03822},
archivePrefix = {arXiv},
       eprint = {1806.03822},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

class Squad2(Squad):
    """SQUAD: The Stanford Question Answering Dataset. Version 2.0."""
    _URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    _DEV_FILE = "dev-v2.0.json"
    _TRAINING_FILE = "train-v2.0.json"

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "id":
                    tf.string,
                "title":
                    tfds.features.Text(),
                "context":
                    tfds.features.Text(),
                "question":
                    tfds.features.Text(),
                "answers":
                    tfds.features.Sequence({
                        "text": tfds.features.Text(),
                        "answer_start": tf.int32,
                    }),
            }),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )