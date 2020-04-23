import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


def main():
    test = tfds.load(
            "squad", data_dir="/tf/data/tf_data", split='validation')
    print(next(test.as_numpy_iterator()))


if __name__ == "__main__":
    main()
