from tensorflow.keras.layers import Dense


class FixedDense(Dense):
    def __init__(self, units, weights, **kwargs):
        super().__init__(
            units,
            weights=weights,
            trainable=False,
            **kwargs
        )
