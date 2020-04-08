"""
Class for handling Inception 5h Model.
"""
import tensorflow as tf

import os

from .common import download
from .core import Model

class Inception5hModel(Model):
    """
    The Inception 5h Model trained to classify images into 1000 categories. 
    """
    input_name = "input:0"
    layers_names = [
        "conv2d0:0",
        "conv2d1:0",
        "conv2d2:0",
        "mixed3a:0",
        "mixed3b:0",
        "mixed4a:0",
        "mixed4b:0",
        "mixed4c:0",
        "mixed4d:0",
        "mixed4e:0",
        "mixed5a:0",
        "mixed5b:0",
        "output2:0",
    ]

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
        self.layers = [
                self.graph.get_tensor_by_name(name) for name in self.layers_names
            ]
