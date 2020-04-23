import numpy as np
import tensorflow as tf

from internn.common import scale_to_image
import cv2
import scipy as sp


class CAM:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        layer_tensor,
        output_tensor,
        weights_tensor,
        input_image,
        blend=0.5,
        colormap=cv2.COLORMAP_JET,
        pred=None,
    ):
        """
        NOTES: weights_tensor.shape[1] == layer_tensor[0]
        """
        input_image_expanded = np.expand_dims(input_image, axis=0)
        feed_dict = {self.model.input_name: input_image_expanded}

        sess = tf.compat.v1.get_default_session()
        layer_output, output = sess.run([layer_tensor, output_tensor], feed_dict)
        layer_output, output = layer_output[0], output[0]

        pred = np.argmax(output) if pred is None else pred

        weights = weights_tensor[:, pred]
        cam = np.dot(layer_output, weights)

        height, width, _ = input_image.shape
        cam = cv2.resize(cam, (width, height))
        cam = cv2.applyColorMap(scale_to_image(cam), colormap)
        cam = cv2.addWeighted(cam, 1 - blend, input_image, blend, 0)

        return cam
