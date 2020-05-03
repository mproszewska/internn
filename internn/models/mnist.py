"""
Handling MNIST digits classification Model.
"""
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import os

from .core import Model


class MNISTModel(Model):
    """
    MNIST digits classification Model trained to classify 10 digits. 
    """

    def __init__(self, force_creation=False):
        self.model_dir = "data/mnist"
        self.graph_def_path = "saved_model.pb"

        file_path = os.path.join(self.model_dir, self.graph_def_path)

        if force_creation or not os.path.exists(file_path):
            create_and_save_model(self.model_dir)

        self.graph = tf.compat.v1.get_default_graph()

        with self.graph.as_default():
            model = build_model()
            model.set_weights(tf.keras.models.load_model(self.model_dir).get_weights())

            self.input = model.input
            self.output = model.output

            self.input_name = self.input.name
            self.output_name = self.output.name

            self.conv_layers, self.conv_layers_names, self.output = load_conv_layers(
                model
            )

    def create_feed_dict(self, input_image):
        """
        Creates and returns feed dict for MNIST model with input image.

        Returns
        -------
        dict
            Feed dict with image

        """
        normalized_input_image = input_image.astype("float32") / 255.0
        normalized_input_image = normalized_input_image.reshape(self.input.shape[1:])
        return super().create_feed_dict(normalized_input_image)


def load_conv_layers(model):
    """
    Finds all Conv2D layers and their names in model.

    Parameters
    ----------
    model : tensorflow Model
        Model with layers.

    Returns
    -------
    layers : list of Tensors
        List of Tensors that represent Conv2D layers.
    layers_names : list of str
        List of Tensor's names.

    """
    conv_layers, conv_layers_names = list(), list()
    x = model.input
    for layer in model.layers:
        x = layer(x)
        if "conv2" in x.name:
            conv_layers.append(x)
            conv_layers_names.append(x.name)
    return conv_layers, conv_layers_names, x


def build_model():
    """
    Defines sequential model.

    Returns
    -------
    Tensorflow Model
        Created model.

    """
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                28, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(28, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )


def create_and_save_model(model_dir):
    """
    Builds, trains and saves MNIST model.

    Parameters
    ----------
    model_dir : str
        Directory in which model will be saved.
    """
    print("Creating model.")
    (train_set, test_set), info_set = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_set = train_set.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_set = train_set.cache()
    train_set = train_set.shuffle(info_set.splits["train"].num_examples)
    train_set = train_set.batch(512)
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)

    test_set = test_set.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_set = test_set.batch(512)
    test_set = test_set.cache()
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    model = build_model()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"],
    )

    model.fit(train_set, epochs=5, validation_data=test_set)

    model.save(model_dir)
    print("Model saved: {}.".format(model_dir))
