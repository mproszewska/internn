import sys
sys.path.append("..")

import internn as inn

import cv2
import tensorflow as tf
import numpy as np

model = inn.models.inception5h.Inception5hModel()
session = tf.compat.v1.InteractiveSession(graph=model.graph)

# input_image = cv2.imread("sky.jpeg", cv2.IMREAD_COLOR)
input_image = np.random.rand(300, 300, 3) * 255.0

reporter = inn.Reporter(True)

featureVisualisation = inn.visualization.FeatureVisualization(model, reporter=reporter)

output_image = featureVisualisation(
    model.layers[6],
    input_image=input_image,
    num_epochs=3,
    steps_per_octave=30,
    step_size=5.0,
    tile_size = 512,
    octave_scale=0.7,
    num_octaves_per_epoch=10,
    blend=0.2,
    norm=2,
    op="mean",
    gradient_ascent="smooth",
    tiles="shift"
)

image_save = cv2.imwrite("output/noise.png",output_image)

session.close()
