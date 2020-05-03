"""
Visualizing activations.
"""

import cv2
import tensorflow as tf

from .core import Interpretation
from ..common import create_heatmap, squeeze_into_2D


class ActivationVisualization(Interpretation):
    """
    Class for visualizing activations.
    """

    def __call__(
        self,
        activation_tensor,
        input_image,
        squeeze_op="max",
        interpolation=cv2.INTER_LANCZOS4,
        colormap=cv2.COLORMAP_JET,
        blend=0.5,
    ):
        """
        Visualizes activations by mapping activations into 2D array.

        Parameters
        ----------
        activation_tensor : Tensor or list of Tensors
            Activations to visualize.
        input_image : ndarray
            Image for which activations will be calculated.
        squeeze_op : str, optional
            Operation used to map activations into 2D array. Accepted values are: "max", "min", 
            "mean". The default is "max".
        interpolation : cv2 interpolation type, optional
            Parameter used to resize activations array to input_image's size. The default is 
            cv2.INTER_LANCZOS4.
        colormap : cv2 colorMap, optional
            Colormap for visualizing activations. The default is COLORMAP_JET.    
        blend : float between 0.0 and 1.0, optional
            Blend factor for combining input_image with calculated activations. Setting blend to
            1.0 would result in returning not changed input_image.         

        Returns
        -------
        ndarray or list of ndarrays
            Images that represent calculated activations for each activation Tensor.

        """
        if blend < 0.0 or blend > 1.0:
            raise ValueError("Invalid blend value {}".format(blend))

        if self.__class__.__name__ == "ActivationVisualization":
            self.reporter.report_parameters({"class_name": self.__class__.__name__})

        params = {
            "squeeze_op": squeeze_op,
            "interpolation": interpolation,
            "colormap": colormap,
            "blend": blend,
        }
        self.reporter.report_parameters(params)

        if isinstance(activation_tensor, tf.Tensor):
            activation_tensor_list = [activation_tensor]
        else:
            activation_tensor_list = activation_tensor

        feed_dict = self.model.create_feed_dict(input_image)

        sess = tf.compat.v1.get_default_session()
        activations_list = sess.run(activation_tensor_list, feed_dict)

        activations_list = [activation[0] for activation in activations_list]
        activations_list = [
            squeeze_into_2D(activation) for activation in activations_list
        ]

        activations_heatmaps_list = list()
        for i, activation in enumerate(activations_list):
            heatmap = create_heatmap(
                activation,
                input_image,
                interpolation=interpolation,
                colormap=colormap,
                blend=blend,
            )
            activations_heatmaps_list.append(heatmap)
            self.plotter.plot_image(heatmap, "{}{}".format(self.__class__.__name__, i))

        if isinstance(activation_tensor, tf.Tensor):
            return activations_heatmaps_list[0]
        else:
            return activations_heatmaps_list


class LayerActivationVisualization(ActivationVisualization):
    """
    Class for visualizing layer's activations.
    """

    def __call__(
        self,
        layer_name,
        input_image,
        squeeze_op="max",
        interpolation=cv2.INTER_LANCZOS4,
        colormap=cv2.COLORMAP_JET,
        blend=0.5,
    ):
        """
        Visualizes activations by mapping activations in layer into 2D array.

        Parameters
        ----------
        layer_name : str
            Name of layer with activations to visualize.
        input_image : ndarray
            Image for which activations will be calculated.
        squeeze_op : str, optional
            Operation used to map activations into 2D array. Accepted values are: "max", "min", 
            "mean". The default is "max".
        interpolation : cv2 interpolation type, optional
            Parameter used to resize activations array to input_image's size. The default is 
            cv2.INTER_LANCZOS4.
        colormap : cv2 colorMap, optional
            Colormap for visualizing activations. The default is COLORMAP_JET.    
        blend : float between 0.0 and 1.0, optional
            Blend factor for combining input_image with calculated activations. Setting blend to
            1.0 would result in returning not changed input_image.         

        Returns
        -------
        ndarray or list of ndarrays
            Images that represent calculated activations in layer.

        """
        if self.__class__.__name__ == "LayerActivationVisualization":
            params = {
                "class_name": self.__class__.__name__,
                "layer_name": layer_name,
            }
            self.reporter.report_parameters(params)

        layer_tensor = self.model.find_layer_tensor(layer_name)

        return super().__call__(
            activation_tensor=layer_tensor,
            input_image=input_image,
            squeeze_op=squeeze_op,
            interpolation=interpolation,
            colormap=colormap,
            blend=blend,
        )
