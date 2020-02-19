import tensorflow as tf


class DummyGenerator:
    def __init__(self, input_shape, output_shape):
        self.model = tf.keras.models.Sequential()
