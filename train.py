import tensorflow as tf
from text_gan.data.qgen_data import QuestionContextPairs, CONFIG
from text_gan.models.squad_qgan import get_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = QuestionContextPairs.load(CONFIG.SAVELOC)

train = data.train.batch(1).prefetch(2)
model = get_model(CONFIG, 1e-3)
model.fit(train, epochs=10)
model.save(CONFIG.MODELSAVELOC)
