"""
Implementation of algorithm that creates saliency map from gradients.
"""
import cv2
import numpy as np
import tensorflow as tf

from .core import Interpretation
from ..common import create_gradient, create_heatmap, squeeze_into_2D


class Gradient(Interpretation):
    """
    Class with algorithm implementation.
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
        Creates saliency map based on gradients.

        Parameters
        ----------
        xs_tensor : Tensor
            Tensor with respect to which the gradient is calculated.
        loss_tensor : Tensor
            Target tensor whose gradient is calculated.
        input_image : ndarray
            Image on which algorithm is performed.
        relu : bool, optional
            Bool that determines if relu is performed on occlusion output. The default is False.
        squeeze_op : str, optional
            Operation used to map values on axis into one value. Acceptable values are: "max", 
            "min", "mean". The default is "max". 
        interpolation : cv2 interpolation type, optional
            Parameter used to resize result to input_image's size. The default is 
            cv2.INTER_LANCZOS4.
        colormap : cv2 colorMap, optional
            Colormap for visualizing result. The default is COLORMAP_JET.    
        blend : float between 0.0 and 1.0, optional
            Blend factor for combining input_image with calculated results. Setting blend to
            1.0 would result in returning not changed input_image.    
        loss_abs : bool, optional
            Determines if activation values or their absolute values should be combined. The 
            default is False.
        loss_op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
             "max", "min", "std".

        Returns
        -------
        ndarray
            Result as image.

        """
        if self.__class__.__name__ == "SaliencyMap":
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

        gradient_func = create_gradient(xs_tensor, loss_tensor, loss_abs, loss_op,)

        feed_dict = self.model.create_feed_dict(input_image)

        sess = tf.compat.v1.get_default_session()
        gradient = sess.run(gradient_func, feed_dict)
        if relu:
            gradient = np.maximum(0, gradient)
        saliency_map = squeeze_into_2D(gradient, op=squeeze_op)

        heatmap = create_heatmap(
            saliency_map,
            input_image,
            interpolation=interpolation,
            colormap=colormap,
            blend=blend,
        )

        self.plotter.plot_image(heatmap, self.__class__.__name__)

        return heatmap
