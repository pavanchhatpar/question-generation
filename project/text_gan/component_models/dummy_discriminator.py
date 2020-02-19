import tensorflow as tf


class DummyDiscriminator:
    def __init__(self):
        self.model = tf.keras.models.Sequential()
