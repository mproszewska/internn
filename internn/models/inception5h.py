"""
Handling Inception 5h Model.
"""
import tensorflow as tf
import os
import re
import sys
import tarfile
import zipfile

from pathlib import Path
from urllib.request import urlretrieve

from .core import Model


class Inception5hModel(Model):
    """
    The Inception 5h Model trained to classify images into 1000 categories. 
    """

    def __init__(self, force_creation=False):
        model_dir = "data/inception5h"
        graph_def_path = "tensorflow_inception_graph.pb"

        self.input_name = "input"
        self.output_name = "output2"

        url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
        filename = url.split("/")[-1]
        file_path = os.path.join(model_dir, filename)

        if force_creation or not os.path.exists(file_path):
            download_model(url=url, dest_dir=model_dir, file_path=file_path)

        path = os.path.join(model_dir, graph_def_path)

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

        self.input = self.graph.get_tensor_by_name("{}:0".format(self.input_name))
        self.conv_layers, self.conv_layers_names = load_conv_layers(self.graph)
        self.output = self.graph.get_tensor_by_name("{}:0".format(self.output_name))


def load_conv_layers(graph):
    """
    Finds all Conv2D layers and their names in graph.

    Parameters
    ----------
    graph : Graph
        Graph with model.

    Returns
    -------
    layers : list of Tensors
        List of Tensors that represent Conv2D layers.
    layers_names : list of str
        List of Tensor's names.

    """
    conv_layers_names = [
        op.name for op in graph.get_operations() if op.type == "Conv2D"
    ]
    conv_layers = list()

    for i, layer in enumerate(conv_layers_names):
        layer_name = re.search("(.+?)/conv", layer).group(1)
        conv_layers.append(graph.get_tensor_by_name("{}:0".format(layer_name)))
        

    return conv_layers, conv_layers_names


def download_model(url, dest_dir, file_path):
    """
    Download and extract the data if it doesn't already exist.

    PARAMETERS
    ----------
    url : str
        URL for to download tar-file.
    dest_dir : str, optional
        Destination folder for the downloaded file.
        
    """

    def download_progress_hook(count, blockSize, totalSize):
        percent = float(count * blockSize / totalSize)
        sys.stdout.write("\rDownload progress: {0:.0%}".format(percent))
        sys.stdout.flush()

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    new_file_path, _ = urlretrieve(
        url=url, filename=file_path, reporthook=download_progress_hook
    )

    print("\nDownload finished. Extracting files.")

    if new_file_path.endswith((".tar.gz", ".tgz")):
        tarfile.open(name=file_path, mode="r:gz").extractall(dest_dir)
    elif new_file_path.endswith(".zip"):
        zipfile.ZipFile(file=file_path, mode="r").extractall(dest_dir)
    else:
        print("Cannot extract. Unsupported extension of downloaded file.")
        return

    print("Done.")
