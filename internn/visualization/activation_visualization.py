import numpy as np
import tensorflow as tf

from internn.common import scale_to_image, squeeze_2D


class ActivationVisualization:
    
    def __init__(self, model):
        self.model = model
        
    def __call__(
        self, activation_tensor, input_image, squeeze_op="max"
    ):
        if isinstance(activation_tensor, tf.Tensor):
            activation_tensor_list = [activation_tensor]
        else:
            activation_tensor_list = activation_tensor
            
        input_image_expanded = np.expand_dims(input_image, axis=0)
        feed_dict = {self.model.input_name: input_image_expanded}

        sess = tf.compat.v1.get_default_session()
        activations_list = sess.run(activation_tensor_list, feed_dict)
        activations_list = [activation[0] for activation in activations_list]
        activations_list = [squeeze_2D(activation) for activation in activations_list]
        activations_list = [scale_to_image(activation) for activation in activations_list]
        
        if isinstance(activation_tensor, tf.Tensor):
            return activations_list[0]
        else:
            return activations_list
        
        

    