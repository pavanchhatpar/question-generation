import tensorflow as tf
import numpy as np
import ujson as json

from text_gan.data import QuestionContextPairs, CONFIG
from text_gan.models import AttnGen

# tf.debugging.set_log_device_placement(True)


def main():
    data = QuestionContextPairs.load(CONFIG.SAVELOC)
    train = data.train.batch(32)
    to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
    train = train.apply(to_gpu)
    with tf.device("/gpu:0"):
        train = train.prefetch(2)
    word_emb_mat = np.array(json.load(open(CONFIG.WORDEMBMAT, "r")))
    qword_emb_mat = np.load(f"{CONFIG.QWORDEMBMAT}.npy")
    qword2idx = json.load(open(CONFIG.QWORD2IDX, "r"))
    idx2qword = np.full(CONFIG.QVOCAB_SIZE, "<UNK>", dtype='object')
    for word, idx in qword2idx.items():
        idx2qword[idx] = word
    model = AttnGen(word_emb_mat, qword_emb_mat, qword2idx, idx2qword, CONFIG)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "/tf/data/attngen/checkpoint/ckpt.{epoch:03d}.tf",
            monitor='loss', verbose=1, save_weights_only=True)
    ]
    model.model.compile(
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    model.fit(train, epochs=100, callbacks=callbacks)
    model.save("/tf/data/attngen/model/model.tf")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    main()
