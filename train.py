import tensorflow as tf
from text_gan.data.qgen_data import QuestionContextPairs, CONFIG
from text_gan.models.squad_qgan import QGAN

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

data = QuestionContextPairs.load(CONFIG.SAVELOC)

train = data.train.batch(2)
to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
train = train.apply(to_gpu)
with tf.device("/gpu:0"):
    train = train.prefetch(2)
model = QGAN(CONFIG, 1e-3)
model.fit(train, epochs=10)
model.save()
