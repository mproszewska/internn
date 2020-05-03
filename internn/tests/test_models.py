import pytest

import numpy as np

import internn as inn


@pytest.mark.parametrize(
    "Model, layers, shape",
    [(inn.Inception5hModel, 59, (10, 100, 3)), (inn.MNISTModel, 2, (28, 28, 1))],
)
def test_model(Model, layers, shape):
    model = Model()
    sess = model.start_session()

    assert model.output is not None
    assert model.input is not None
    assert len(model.conv_layers) == layers
    assert len(model.conv_layers_names) == layers

    with pytest.raises(ValueError):
        model.find_layer_tensor("test")
    assert model.input == model.find_layer_tensor(model.input_name)

    layer_name = model.conv_layers_names[1]
    assert model.find_neuron_tensor(layer_name, 7).shape[-1] == 1

    feed_dict = model.create_feed_dict(np.zeros(shape))
    assert len(feed_dict) == 1
    sess.close()


def test_mnist_feed_dict():
    model = inn.MNISTModel()
    input_image = np.random.uniform(0, 255, size=(28, 28))
    feed_dict = model.create_feed_dict(input_image)
    result = feed_dict[model.input_name]
    assert result.max() <= 1.0 and result.min() >= 0
    assert result.dtype == np.float32
