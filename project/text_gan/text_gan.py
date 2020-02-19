import tensorflow as tf


class TextGan:
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        self.model = tf.keras.models.Sequential()
        self.model.add(generator)
        self.model.add(discriminator)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=opt
        )
