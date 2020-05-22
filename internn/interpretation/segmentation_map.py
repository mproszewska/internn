import numpy as np

"""
Implementation of Segmentation Map with Graph-Cut
"""
import cv2
import maxflow

from .core import Interpretation
from ..common import create_heatmap, squeeze_into_2D


class SegmentationMap(Interpretation):
    """
    Class with Segmentation Map with Graph-Cut implementation.
    """

    def __call__(
        self,
        saliency_map,
        input_image,
        bg_threshold=0.3,
        fg_threshold=0.95,
        weight_norm=2,
        squeeze_op="max",
        interpolation=cv2.INTER_LANCZOS4,
        colormap=cv2.COLORMAP_JET,
        blend=0.0,
    ):
        """
        Creates Segmentation Map based od saliency map using Graph-Cut algorithm.

        Parameters
        ----------
        saliency_map : ndarray
            2D ndarray that represents saliency map for input_image. Could be obtained by running
            on of interpretation algorithms with cv2.COLORMAP_BONE colormap and blend set to 0. 
            Obtained like that heatmap has to be squeezed into 2D array.
        input_image : ndarray
            Image on which algorithm is performed.
        bg_threshold : float from [0,1], optional
            Threshold for background objects. The default is 0.3.
        fg_threshold : float from [0,1], optional
           Threshold for foreground objects. The default is 0.95.
        weight_norm : int of "inf" optional
            Power to which input_image value is raised during weight calculation. The default is 2.
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
            1.0 would result in returning not changed input_image. The default is 0.

        Returns
        -------
        ndarray
            Result as image.

        """
        if self.__class__.__name__ == "SegmentationMap":
            params = {
                "class_name": self.__class__.__name__,
                "bg_threshold": bg_threshold,
                "fg_threshold": fg_threshold,
                "weight_norm": weight_norm,
                "squeeze_op": squeeze_op,
                "interpolation": interpolation,
                "colormap": colormap,
                "blend": blend,
            }
            self.reporter.report_parameters(params)

        saliency_map = squeeze_into_2D(saliency_map, op=squeeze_op)

        img_classification = self.run_img_classification(
            saliency_map, bg_threshold, fg_threshold
        )
        segments_classification = self.run_segments_classification(
            img_classification, input_image, weight_norm
        )

        heatmap = create_heatmap(
            segments_classification,
            input_image,
            interpolation=interpolation,
            colormap=colormap,
            blend=blend,
        )

        self.plotter.plot_image(heatmap, self.__class__.__name__)

        return heatmap

    def run_img_classification(self, gradient, bg_threshold=0.3, fg_threshold=0.9):
        bg_q = np.quantile(gradient, q=bg_threshold)
        fg_q = np.quantile(gradient, q=fg_threshold)

        classification = np.zeros_like(gradient)
        not_bg_class = gradient > bg_q
        fg_class = gradient >= fg_q
        classification = 0.5 * (not_bg_class.astype("int16") + fg_class.astype("int16"))
        return classification

    def edge_weight(
        self, input_image, w, h, w_other, h_other, weight_norm=2
    ):
        diff = np.absolute(input_image[h, w] - input_image[h_other, w_other])
        normed = np.power(diff, weight_norm)
        if weight_norm == "inf":
            return 1 / (1 + np.max(normed))
        else:
            return 1 / (1 + np.sum(normed))

    def run_segments_classification(
        self, img_classification, input_image, weight_norm=2
    ):
        num_nodes = int(img_classification.size + 2)
        h_max, w_max = img_classification.shape

        graph = maxflow.Graph[float](h_max * w_max, (h_max - 1) * (w_max - 1) * 3)
        graph_nodes = graph.add_nodes(num_nodes)

        for (h, w), value in np.ndenumerate(img_classification):
            node_id = h * w_max + w
            if value == 0.0:
                node = [node_id, num_nodes, 0]
            elif value == 1.0:
                node = [node_id, 0, num_nodes]
            else:
                node = [node_id, 0, 0]
            graph.add_tedge(graph_nodes[node[0]], node[1], node[2])

        for (h, w), value in np.ndenumerate(img_classification):
            if h + 1 == h_max or w + 1 == w_max:
                continue
            node_id = h * w_max + w

            h_other, w_other = h + 1, w
            other_id = h_other * w_max + w_other
            weight = self.edge_weight(
                input_image, w, h, w_other, h_other, weight_norm
            )
            graph.add_edge(node_id, other_id, weight, weight)

            h_other, w_other = h, w + 1
            other_id = h_other * w_max + w_other
            weight = self.edge_weight(
                input_image, w, h, w_other, h_other, weight_norm
            )
            graph.add_edge(node_id, other_id, weight, weight)

            h_other, w_other = h + 1, w + 1
            other_id = h_other * w_max + w_other
            weight = self.edge_weight(
                input_image, w, h, w_other, h_other, weight_norm
            )
            graph.add_edge(node_id, other_id, weight, weight)

        graph.maxflow()

        segments_classification = np.empty_like(img_classification)
        for node_id in range(img_classification.size):
            h = node_id // w_max
            w = node_id - h * w_max
            segments_classification[h, w] = graph.get_segment(graph_nodes[node_id])

        return segments_classification
