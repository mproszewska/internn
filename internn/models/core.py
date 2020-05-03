"""
Handling model.
"""
import numpy as np
import tensorflow as tf

import os
import re


class Model:
    """
    Base class that each model extends.
    """

    def __init__(self):
        """
        Sets up model.
        
        """
        raise NotImplementedError()

    def classify(self):
        """
        Returns final activation for each class.

        Returns
        -------
        Tensor
            Tensor with activation for each class.

        """
        return tf.reduce_max(self.output, axis=0)

    def create_feed_dict(self, input_image):
        """
        Creates and returns feed dict for model with input image.

        Returns
        -------
        dict
            Feed dict with image

        """
        input_name = (
            self.input_name
            if ":" in self.input_name
            else "{}:0".format(self.input_name)
        )
        input_image_expanded = np.expand_dims(input_image, axis=0)
        feed_dict = {input_name: input_image_expanded}
        return feed_dict

    def find_layer_tensor(self, layer_name):
        """
        Finds layer tensor in model by name.

        Parameters
        ----------
        layer_name : str
            Name of the layer.

        Raises
        ------
        ValueError
            If layer_name is invalid.

        Returns
        -------
        Tensor
            Tensor that represents layer.

        """
        if self.input_name == layer_name:
            return self.input
        if self.output_name == layer_name:
            return self.output

        layer_id = self.conv_layers_names.index(layer_name)
        return self.conv_layers[layer_id]

    def find_neuron_tensor(self, layer_name, neuron_num):
        """
        Finds tensor that represents neuron in model.

        Parameters
        ----------
        layer_name : str
            Name of the layer.
        neuron_num : int
            Neuron's number in layer.

        Raises
        ------
        ValueError
            If layer_name is invalid.

        Returns
        -------
        Tensor
            Tensor that represents neuron.

        """
        layer = self.find_layer_tensor(layer_name)

        neuron_begin = np.zeros(len(layer.shape), dtype=np.int32)
        neuron_begin[-1] = neuron_num

        neuron_size = -np.ones(len(layer.shape), dtype=np.int32)
        neuron_size[-1] = 1

        return tf.slice(layer, begin=neuron_begin, size=neuron_size)

    def start_session(self):
        """
        Starts session.

        Returns
        -------
        session : Tensorflow Session
            Started session.

        """
        session = tf.compat.v1.InteractiveSession(graph=self.graph)
        init = tf.compat.v1.global_variables_initializer()
        session.run(init)
        return session
