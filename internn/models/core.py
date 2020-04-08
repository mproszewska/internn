"""
Class for handling model.
"""
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