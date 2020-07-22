from transformers import pipeline
import tensorflow as tf
from logging import Formatter
from absl import app, logging, flags
import os
import json
from tqdm.autonotebook import tqdm

from text_gan.utils import SQuADReader
from text_gan import cfg, cfg_from_file


FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", None, "Config YAML filepath")
flags.DEFINE_string("set", None, "train/dev set")
flags.DEFINE_string("predicted", None, "json file with predicted questions")
flags.DEFINE_string("out", None, "Output files' prefix")


def main(_):
    if FLAGS.cfg is not None:
        cfg_from_file(FLAGS.cfg)

    if FLAGS.log_dir is not None and FLAGS.log_dir != "":
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        if not os.path.isdir(FLAGS.log_dir):
            raise ValueError(f"{FLAGS.log_dir} should be a directory!")
        logging.get_absl_handler().use_absl_log_file()

    logging.get_absl_handler().setFormatter(
        Formatter(fmt="%(levelname)s:%(message)s"))

    if FLAGS.set is None or FLAGS.set not in ['train', 'dev']:
        raise ValueError("Choose a set, train/ dev")

    FLAGS.set = (
        cfg.RAW_TRAIN_SAVE if FLAGS.set == 'train' else cfg.RAW_DEV_SAVE)

    if FLAGS.out is None:
        raise ValueError("Give an output filename prefix")

    reader = SQuADReader()
    data = reader.flatten_parsed(reader.parse(FLAGS.set, qids=True), qids=True)

    with open(FLAGS.predicted, "r") as fp:
        preds = json.load(fp)

    nlp = pipeline("question-answering")

    pred_ans = {}
    orig_ans = {}

    batch_size = 256

    batch_context = []
    batch_pred_question = []
    batch_orig_question = []
    batch_qid = []

    for sample in tqdm(data):
        pred_question = preds.get(sample['qid'], None)
        if not pred_question:
            continue
        pred_question = pred_question[0]
        pred_question = pred_question[:pred_question.find("EOS")]
        batch_context.append(sample['context'])
        batch_pred_question.append(pred_question)
        batch_orig_question.append(sample['question'])
        batch_qid.append(sample['qid'])

        if len(batch_context) == batch_size:
            try:
                answers = nlp(
                    context=batch_context,
                    question=batch_pred_question)['answer']
                for answer, qid in answers, zip(batch_qid):
                    pred_ans[qid] = answer
            except Exception as e:  # noqa
                pass

            try:
                answers = nlp(
                    context=batch_context,
                    question=batch_orig_question)['answer']
                for answer, qid in answers, zip(batch_qid):
                    orig_ans[qid] = answer
            except Exception as e:  # noqa
                pass

            batch_context = []
            batch_pred_question = []
            batch_orig_question = []
            batch_qid = []

    if len(batch_context) > 0:
        try:
            answers = nlp(
                context=batch_context,
                question=batch_pred_question)['answer']
            for answer, qid in answers, zip(batch_qid):
                pred_ans[qid] = answer
        except Exception as e:  # noqa
            pass

        try:
            answers = nlp(
                context=batch_context,
                question=batch_orig_question)['answer']
            for answer, qid in answers, zip(batch_qid):
                orig_ans[qid] = answer
        except Exception as e:  # noqa
            pass

    with open(f"{FLAGS.out}-pred-ans.json", "w") as fp:
        json.dump(pred_ans, fp)

    with open(f"{FLAGS.out}-orig-ans.json", "w") as fp:
        json.dump(orig_ans, fp)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    app.run(main)
