"""
Class for handling model.
"""
import numpy as np
import tensorflow as tf


class Model:
    """
    Base class that each model extends.
    """

    def __init__(self):
        """
        Sets up model.
        """
        raise NotImplementedError

    def loss(self, activation_tensor, norm=2, op="mean"):
        """
        Returns norm of neuron activations in given tensor.

        PARAMETERS
        ----------
        activation_tensor : Tensor
            Tensor that contains neurons which norm is calculated.
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std". The default is "mean".
            
        RETURNS
        -------
        float
            Norm of neuron activations.
        """
        norm = tf.constant([norm], dtype=activation_tensor.dtype)
        activation_tensor = tf.math.pow(x=activation_tensor, y=norm)

        if op == "mean":
            loss_result = tf.math.reduce_mean(input_tensor=activation_tensor)
        elif op == "max":
            loss_result = tf.math.reduce_max(input_tensor=activation_tensor)
        elif op == "min":
            loss_result = tf.math.reduce_max(input_tensor=activation_tensor)
        elif op == "std":
            loss_result = tf.math.reduce_std(input_tensor=activation_tensor)
        else:
            raise ValueError("Unsupported opperation: {}".format(op))

        return loss_result

    def find_layer_tensor(self, layer_name):
        if self.input_name == layer_name:
            return self.input
        if self.output_name == layer_name:
            return self.output
        layer_id = self.conv_layers_names.index(layer_name)
        return self.conv_layers[layer_id]

    def find_neuron_tensor(self, layer_name, neuron_num):
        layer = self.find_layer_tensor(layer_name)

        neuron_begin = np.zeros(len(layer.shape), dtype=np.int32)
        neuron_begin[-1] = neuron_num
        neuron_size = -np.ones(len(layer.shape), dtype=np.int32)
        neuron_size[-1] = 1
        neuron_tensor = tf.slice(layer, begin=neuron_begin, size=neuron_size)

        return neuron_tensor
