import pytest

import numpy as np

import internn as inn
from ..common import activation_loss, squeeze_into_2D, scale_to_image


@pytest.mark.parametrize(
    "Model, size",
    [(inn.Inception5hModel, (200, 200, 3)), (inn.MNISTModel, (28, 28, 1))],
)
@pytest.mark.parametrize("abs, op", [(False, "max"), (True, "mean")])
def test_activation_loss(Model, size, abs, op):
    model = Model()
    sess = model.start_session()

    input_image = np.random.normal(0.0, 1.0, size=size)
    feed_dict = model.create_feed_dict(input_image)

    activation_loss_func = activation_loss(model.conv_layers[0], abs=abs, op=op)
    result = sess.run(activation_loss_func, feed_dict)
    assert isinstance(result, np.float32)
    assert result != 0.0
    sess.close()


def test_squeeze_into_2D():
    array = np.arange(8).reshape((2, 2, 2))
    array_squeezed = squeeze_into_2D(array, op="mean")
    expected = np.array([[0.5, 2.5], [4.5, 6.5]])
    assert np.array_equal(array_squeezed, expected)

    array_squeezed = squeeze_into_2D(array, op="max")
    expected = np.array([[1, 3], [5, 7]])
    assert np.array_equal(array_squeezed, expected)

    array_squeezed = squeeze_into_2D(array, op="min")
    expected = np.array([[0, 2], [4, 6]])
    assert np.array_equal(array_squeezed, expected)


def test_scale_to_image():
    array = np.arange(8)
    array_scaled = scale_to_image(array)
    expected = np.array([0, 36, 72, 109, 145, 182, 218, 255], dtype=np.uint8)
    assert np.array_equal(array_scaled, expected)

    array = np.array([0.23, 0.4, 1.7])
    array_scaled = scale_to_image(array)
    expected = np.array([0, 29, 255], dtype=np.uint8)
    assert np.array_equal(array_scaled, expected)
