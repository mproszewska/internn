import numpy as np
import tensorflow as tf

import maxflow

from internn.plot import Plotter
from internn.report import Reporter


def grad_to_saliency_map(gradient, map_op="max"):
    if len(gradient.shape) == 2:
        return gradient
    if map_op == "max":
        return gradient.max(axis=2)
    if map_op == "mean":
        return gradient.mean(axis=2)
    raise ValueError("Invalid map_op value: {}".format(map_op))


def scale_to_image(array):
    image = (255.0 * (array - array.min()) / (array.max() - array.min())).astype(
        "uint8"
    )
    return image


def run_gradient(model, feature_tensor, input_image, norm=2, norm_op="mean"):
    sess = tf.compat.v1.get_default_session()
    loss_func = model.loss(feature_tensor, norm=norm, op=norm_op)
    gradient_func = tf.gradients(ys=loss_func, xs=model.input)[0]

    intput_image_expanded = np.expand_dims(input_image, axis=0)
    feed_dict = {model.input_name: intput_image_expanded}

    gradient = sess.run(gradient_func, feed_dict)[0]
    return gradient


class SaliencyMap:
    def __init__(self, model, reporter=None, plotter=None):
        self.model = model
        self.reporter = Reporter() if reporter is None else reporter
        self.plotter = Plotter() if plotter is None else plotter

    def __call__(
        self, feature_tensor, input_image, norm=2, norm_op="mean", map_op="max"
    ):

        self.reporter.report_saliency_map(
            attribution_name=self.__class__.__name__,
            norm=norm,
            norm_op=norm_op,
            map_op=map_op,
        )

        gradient = run_gradient(self.model, feature_tensor, input_image, norm, norm_op)

        saliency_map = grad_to_saliency_map(gradient, map_op)
        saliency_map_image = scale_to_image(saliency_map)

        return saliency_map_image


def run_bg_fg_classification(gradient, bg_threshold=0.3, fg_threshold=0.9):
    bg_q = np.quantile(gradient, q=bg_threshold)
    fg_q = np.quantile(gradient, q=fg_threshold)

    classification = np.zeros_like(gradient)
    not_bg_class = gradient > bg_q
    fg_class = gradient >= fg_q
    classification = 0.5 * (not_bg_class.astype("int16") + fg_class.astype("int16"))
    return classification


def edge_weight(input_image, w, h, w_other, h_other, weight_norm=2, weight_op="sum"):
    diff = np.absolute(input_image[h, w] - input_image[h_other, w_other])
    normed = np.power(diff, weight_norm)
    if weight_op == "sum":
        return 1 / (1 + np.sum(normed))
    if weight_op == "mean":
        return 1 / (1 + np.mean(normed))
    if weight_op == "min":
        return 1 / (1 + np.min(normed))
    if weight_op == "max":
        return 1 / (1 + np.max(normed))
    raise ValueError("Invalid weight_op value: {}".format(weight_op))


def run_segments_classification(
    bg_fg_classification, input_image, weight_norm=2, weight_op="sum"
):
    num_nodes = int(bg_fg_classification.size + 2)
    h_max, w_max = bg_fg_classification.shape

    nodes, edges = list(), list()
    graph = maxflow.Graph[float](h_max * w_max, (h_max - 1) * (w_max - 1) * 3)
    graph_nodes = graph.add_nodes(num_nodes)

    for (h, w), value in np.ndenumerate(bg_fg_classification):
        node_id = h * w_max + w

        if value == 0.0:
            node = [node_id, num_nodes, 0]
        elif value == 1.0:
            node = [node_id, 0, num_nodes]
        else:
            node = [node_id, 0, 0]

        nodes.append(node)
        graph.add_tedge(graph_nodes[node[0]], node[1], node[2])

    for (h, w), value in np.ndenumerate(bg_fg_classification):
        if h + 1 == h_max or w + 1 == w_max:
            continue
        node_id = h * w_max + w

        h_other, w_other = h + 1, w
        other_id = h_other * w_max + w_other
        weight = edge_weight(
            input_image, w, h, w_other, h_other, weight_norm, weight_op
        )
        graph.add_edge(node_id, other_id, weight, weight)

        h_other, w_other = h, w + 1
        other_id = h_other * w_max + w_other
        weight = edge_weight(
            input_image, w, h, w_other, h_other, weight_norm, weight_op
        )
        graph.add_edge(node_id, other_id, weight, weight)

        h_other, w_other = h + 1, w + 1
        other_id = h_other * w_max + w_other
        weight = edge_weight(
            input_image, w, h, w_other, h_other, weight_norm, weight_op
        )
        graph.add_edge(node_id, other_id, weight, weight)

    graph.maxflow()

    segments_classification = np.empty_like(bg_fg_classification)
    for node_id in range(bg_fg_classification.size):
        h = node_id // w_max
        w = node_id - h * w_max
        segments_classification[h, w] = graph.get_segment(graph_nodes[node_id])

    return segments_classification


class SegmentationMask:
    def __init__(self, model, reporter=None, plotter=None):
        self.model = model
        self.reporter = Reporter() if reporter is None else reporter
        self.plotter = Plotter() if plotter is None else plotter

    def __call__(
        self,
        feature_tensor,
        input_image,
        norm=2,
        norm_op="mean",
        map_op="max",
        bg_threshold=0.3,
        fg_threshold=0.95,
        weight_norm=2,
        weight_op="sum",
    ):
        gradient = run_gradient(self.model, feature_tensor, input_image)
        saliency_map = grad_to_saliency_map(gradient, map_op)
        bg_fg_classification = run_bg_fg_classification(
            saliency_map, bg_threshold, fg_threshold
        )
        segments_classification = run_segments_classification(
            bg_fg_classification, input_image, weight_norm, weight_op
        )

        output_image = np.empty_like(input_image)
        for (h, w), value in np.ndenumerate(segments_classification):
            if segments_classification[h, w] == 1:
                output_image[h, w] = [0, 0, input_image[h, w, 2]]
            else:
                output_image[h, w] = [input_image[h, w, 0], 0, 0]

        return output_image
