"""
Implementation of occlusion algorithm.
"""
import cv2
import numpy as np
import tensorflow as tf

from .core import Interpretation
from ..common import create_heatmap


class Occlusion(Interpretation):
    """
    Class with occlusion algorithm implementation.
    """

    def __call__(
        self,
        input_image,
        pred=None,
        tile_size=100,
        tile_value=0,
        tile_step=10,
        colormap=cv2.COLORMAP_JET,
        blend=0.5,
        loss_norm=2,
        loss_op="mean",
    ):
        """
        Performs occlusion on input_image.

        Parameters
        ----------
        input_image : ndarray
            Image on which occlusion is performed.
        pred : int, optional
            Number of expected class. If None pred is class predicted by model. The default is None.  
        tile_size : int, optional
            Size of tile. The default is 100.
        tile_value : int, optional
            Value that fills tile during occlusion step. The default is 0.
        tile_step : int, optional
            Distance between tiles boarders horizontally and vertically. The default is 10.
        colormap : cv2 colorMap, optional
            Colormap for visualizing result. The default is COLORMAP_JET.    
        blend : float between 0.0 and 1.0, optional
            Blend factor for combining input_image with calculated results. Setting blend to 1.0
            would result in returning not changed input_image.  
        loss_norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        loss_op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
             "max", "min", "std".

        Returns
        -------
        ndarray
            Result as image.

        """
        params = {
            "class_name": self.__class__.__name__,
            "pred": pred,
            "tile_size": tile_size,
            "tile_value": tile_value,
            "tile_step": tile_step,
            "colormap": colormap,
            "blend": blend,
            "loss_norm": loss_norm,
            "loss_op": loss_op,
        }
        self.reporter.report_parameters(params)

        feed_dict = self.model.create_feed_dict(input_image)
        sess = tf.compat.v1.get_default_session()

        if pred is None:
            pred = sess.run(self.model.classify(), feed_dict).argmax()

        classification = self.model.classify()
        pred_func = classification[pred] / tf.reduce_sum(classification)
        pred_original = sess.run(pred_func, feed_dict)

        image_height, image_width, _ = input_image.shape
        h = 0

        occlusion = np.zeros(input_image.shape[0:2])
        while h + tile_size <= image_height:
            w = 0
            while w + tile_size <= image_width:
                new_image = input_image.copy()
                new_image[h : h + tile_size, w : w + tile_size, :] = tile_value

                feed_dict = self.model.create_feed_dict(new_image)
                pred_current = sess.run(pred_func, feed_dict)

                if pred_current > pred_original:
                    occlusion[h : h + tile_size, w : w + tile_size] -= 1
                elif pred_current < pred_original:
                    occlusion[h : h + tile_size, w : w + tile_size] += 1

                w += tile_step
            h += tile_step

        heatmap = create_heatmap(
            occlusion, input_image, colormap=colormap, blend=blend,
        )

        self.plotter.plot_image(heatmap, self.__class__.__name__)

        return heatmap
