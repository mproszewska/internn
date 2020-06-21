"""
Implementation of Grad-CAM algorithm.
"""
import cv2
import numpy as np
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
        loss_tensor,
        input_image,
        relu=False,
        squeeze_op="max",
        interpolation=cv2.INTER_LANCZOS4,
        colormap=cv2.COLORMAP_JET,
        blend=0.5,
        loss_abs=False,
        loss_op="mean",
    ):
        """
        Performs Grad-CAM with respect to xs_tensor.

        Parameters
        ----------
        xs_tensor : Tensor
            Tensor with respect to which the gradient is calculated.
        loss_tensor : Tensor
            Target tensor whose gradient is calculated.
        input_image : ndarray
            Image on which Grad-CAM is performed.
        relu : bool, optional
            Bool that determines if relu is performed on occlusion output. The default is False.
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
        loss_abs : bool, optional
            Determines if activation values or their absolute values should be combined. The 
            default is False.
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
            "relu": relu,
            "squeeze_op": squeeze_op,
            "interpolation": interpolation,
            "colormap": colormap,
            "blend": blend,
            "loss_abs": loss_abs,
            "loss_op": loss_op,
        }
        self.reporter.report_parameters(params)

        sess = tf.compat.v1.get_default_session()

        gradient_func = create_gradient(
            xs_tensor=xs_tensor,
            loss_tensor=loss_tensor,
            loss_abs=loss_abs,
            loss_op=loss_op,
        )

        feed_dict = self.model.create_feed_dict(input_image)

        sess = tf.compat.v1.get_default_session()
        xs, gradient = sess.run([xs_tensor, gradient_func], feed_dict)
        xs = xs[0]

        grad_cam = np.zeros(shape=gradient.shape[0:2], dtype="float32")

        for i in range(gradient.shape[2]):
            grad_cam += xs[:, :, i] * np.mean(gradient[:, :, i])

        if relu:
            grad_cam = np.maximum(0, grad_cam)

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
