"""
Implementation of Grad-CAM algorithm.
"""
import cv2
import numpy as np
import tensorflow as tf


from .core import Interpretation
from ..common import create_gradient, create_heatmap, squeeze_into_2D


class GradCAM(Interpretation):
    """
    Class with Grad-CAM implementation.
    """

    def __call__(
        self,
        xs_tensor,
        input_image,
        num_classes=10,
        guided_backpropagation=False,
        squeeze_op="max",
        interpolation=cv2.INTER_LANCZOS4,
        colormap=cv2.COLORMAP_JET,
        blend=0.5,
        loss_norm=2,
        loss_op="mean",
    ):
        """
        Performs Grad-CAM with respect to input_image.

        Parameters
        ----------
        xs_tensor : Tensor
            Tensor with respect to which the gradient is calculated.
        input_image : ndarray
            Image on which Grad-CAM is performed. The default is 10.
        num_classes : int
            Number of top predicted classes to consider.
        guided_backpropagation : bool, optional
            Whether to use guided backpropagation. The default is False.
        squeeze_op : str, optional
            Operation used to map values on axis into one value. Acceptable values are: "max", 
            "min", "mean". The default is "max". 
        interpolation : cv2 interpolation type, optional
            Parameter used to resize activations array to input_image's size. The default is 
            cv2.INTER_LANCZOS4.
        colormap : cv2 colorMap, optional
            Colormap for visualizing activations. The default is COLORMAP_JET.    
        blend : float between 0.0 and 1.0, optional
            Blend factor for combining input_image with calculated activations. Setting blend to
            1.0 would result in returning not changed input_image.     
        loss_norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        loss_op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
             "max", "min", "std".

        Returns
        -------
        grad_cam : ndarray
            Grad-CAM result as image.

        """
        params = {
            "class_name": self.__class__.__name__,
            "num_classes": num_classes,
            "guided_backpropagation": guided_backpropagation,
            "squeeze_op": squeeze_op,
            "interpolation": interpolation,
            "colormap": colormap,
            "blend": blend,
            "loss_norm": loss_norm,
            "loss_op": loss_op,
        }
        self.reporter.report_parameters(params)

        gradient_funcs, weights_func = list(), list()

        weights_tensor = self.model.classify()

        feed_dict = self.model.create_feed_dict(input_image)

        sess = tf.compat.v1.get_default_session()
        weights = sess.run(weights_tensor, feed_dict)
        top_classes = weights.argpartition(len(weights) - num_classes)[-num_classes:]

        for num_class in top_classes:
            gradient_func = create_gradient(
                xs_tensor=xs_tensor,
                loss_tensor=weights_tensor[num_class],
                loss_norm=loss_norm,
                loss_op=loss_op,
                guided_backpropagation=guided_backpropagation,
            )
            gradient_funcs.append(gradient_func)
            weights_func.append(weights_tensor[num_class])

        feed_dict = self.model.create_feed_dict(input_image)

        sess = tf.compat.v1.get_default_session()
        gradients, weights = sess.run([gradient_funcs, weights_func], feed_dict)

        grad_cam = np.zeros_like(gradients[0], dtype="float32")
        for grad, weight in zip(gradients, weights):
            grad_cam += grad * weight

        grad_cam = squeeze_into_2D(grad_cam, op=squeeze_op)

        heatmap = create_heatmap(
            grad_cam,
            input_image,
            interpolation=interpolation,
            colormap=colormap,
            blend=blend,
        )

        self.plotter.plot_image(heatmap, self.__class__.__name__)

        return heatmap
