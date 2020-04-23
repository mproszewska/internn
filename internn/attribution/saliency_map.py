import numpy as np
import tensorflow as tf

import maxflow

from internn.common import scale_to_image, squeeze_2D


def run_gradient(
    model,
    layer_tensor,
    input_image,
    norm=2,
    norm_op="mean",
    guided_backpropagation=False,
):
    input_image_expanded = np.expand_dims(input_image, axis=0)
    feed_dict = {model.input_name: input_image_expanded}

    loss_func = model.loss(model.output, norm=norm, op=norm_op)
    gradient_func = tf.gradients(ys=loss_func, xs=layer_tensor)[0]

    sess = tf.compat.v1.get_default_session()
    gradient, layer_output = sess.run([gradient_func, layer_tensor], feed_dict)
    gradient, layer_output = gradient[0], layer_output[0]

    if guided_backpropagation:
        gradient_guided = (gradient > 0).astype(dtype="float32")
        layer_output_guided = (layer_output > 0).astype(dtype="float32")
        gradient_guided = gradient * gradient_guided * layer_output_guided
        return gradient_guided
    else:
        return gradient


class SaliencyMap:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        layer_tensor,
        input_image,
        norm=2,
        norm_op="mean",
        map_op="max",
        guided_backpropagation=False,
    ):
        gradient = run_gradient(
            self.model,
            layer_tensor,
            input_image,
            norm,
            norm_op,
            guided_backpropagation,
        )
        saliency_map = squeeze_2D(gradient, map_op)

        saliency_map_image = scale_to_image(saliency_map)

        return saliency_map_image

    def grad_to_saliency_map(self, gradient, map_op="max"):
        if len(gradient.shape) == 2:
            return gradient
        if map_op == "max":
            return gradient.max(axis=2)
        if map_op == "mean":
            return gradient.mean(axis=2)
        raise ValueError("Invalid map_op value: {}".format(map_op))


class GradientTimesInput:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        input_image,
        norm=2,
        norm_op="mean",
        map_op="max",
        guided_backpropagation=False,
    ):
        gradient = run_gradient(
            self.model,
            self.model.input,
            input_image,
            norm,
            norm_op,
            guided_backpropagation,
        )
        gradient_input = gradient * input_image.astype("float32")

        gradient_input = scale_to_image(gradient_input)

        return gradient_input


class SegmentationMask(SaliencyMap):
    def __call__(
        self,
        input_image,
        norm=2,
        norm_op="mean",
        map_op="max",
        guided_backpropagation=False,
        bg_threshold=0.3,
        fg_threshold=0.95,
        weight_norm=2,
        weight_op="sum",
    ):
        saliency_map = super().__call__(
            layer_tensor=self.model.input,
            input_image=input_image,
            norm=norm,
            norm_op=norm_op,
            map_op=map_op,
            guided_backpropagation=guided_backpropagation,
        )

        img_classification = self.run_img_classification(
            saliency_map, bg_threshold, fg_threshold
        )
        segments_classification = self.run_segments_classification(
            img_classification, input_image, weight_norm, weight_op
        )

        output_image = np.empty_like(input_image)
        for (h, w), value in np.ndenumerate(segments_classification):
            if segments_classification[h, w] == 1:
                output_image[h, w] = [0, 0, input_image[h, w, 2]]
            else:
                output_image[h, w] = [input_image[h, w, 0], 0, 0]

        return output_image

    def run_img_classification(self, gradient, bg_threshold=0.3, fg_threshold=0.9):
        bg_q = np.quantile(gradient, q=bg_threshold)
        fg_q = np.quantile(gradient, q=fg_threshold)

        classification = np.zeros_like(gradient)
        not_bg_class = gradient > bg_q
        fg_class = gradient >= fg_q
        classification = 0.5 * (not_bg_class.astype("int16") + fg_class.astype("int16"))
        return classification

    def edge_weight(
        self, input_image, w, h, w_other, h_other, weight_norm=2, weight_op="sum"
    ):
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
        self, img_classification, input_image, weight_norm=2, weight_op="sum"
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
                input_image, w, h, w_other, h_other, weight_norm, weight_op
            )
            graph.add_edge(node_id, other_id, weight, weight)

            h_other, w_other = h, w + 1
            other_id = h_other * w_max + w_other
            weight = self.edge_weight(
                input_image, w, h, w_other, h_other, weight_norm, weight_op
            )
            graph.add_edge(node_id, other_id, weight, weight)

            h_other, w_other = h + 1, w + 1
            other_id = h_other * w_max + w_other
            weight = self.edge_weight(
                input_image, w, h, w_other, h_other, weight_norm, weight_op
            )
            graph.add_edge(node_id, other_id, weight, weight)

        graph.maxflow()

        segments_classification = np.empty_like(img_classification)
        for node_id in range(img_classification.size):
            h = node_id // w_max
            w = node_id - h * w_max
            segments_classification[h, w] = graph.get_segment(graph_nodes[node_id])

        return segments_classification
