"""
Class for handling Inception 5h Model.
"""
import tensorflow as tf

import os

from .common import download, load_conv_layers, load_weights
from .core import Model

class Inception5hModel(Model):
    """
    The Inception 5h Model trained to classify images into 1000 categories. 
    """
    input_name = "input:0"
    output_name = "output2:0"
    

    def __init__(self):
        """
        Downloads needed data and sets up Inception 5h Model.
        """
        url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
        model_dir = "data/inception5h"
        graph_def_path = "tensorflow_inception_graph.pb"

        download(url=url, dest_dir=model_dir)
        path = os.path.join(model_dir, graph_def_path)
        
        self.graph = tf.Graph()
        with self.graph.as_default():           
            with tf.io.gfile.GFile(path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

        self.input = self.graph.get_tensor_by_name(self.input_name)
        self.conv_layers, self.conv_layers_names = load_conv_layers(self.graph)
        self.output = self.graph.get_tensor_by_name(self.output_name)
        self.weights, self.weights_names = load_weights(graph_def)