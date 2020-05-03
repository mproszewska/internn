import pytest

import numpy as np

import internn as inn


@pytest.mark.parametrize(
    "Model, shape",
    [(inn.Inception5hModel, (200, 100, 3)), (inn.MNISTModel, (28, 28, 1))],
)
@pytest.mark.parametrize("squeeze_op", ["max", "min", "mean"])
def test_activation_visualization(Model, shape, squeeze_op):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.ActivationVisualization(model)

    output_images = interpretator(model.conv_layers, input_image, squeeze_op=squeeze_op)
    assert len(output_images) == len(model.conv_layers)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, layer_num",
    [(inn.Inception5hModel, (200, 100, 3), 40), (inn.MNISTModel, (28, 28, 1), 0)],
)
@pytest.mark.parametrize(
    "squeeze_op, interpolation, colormap, blend",
[("max", 1, 3, 0.2), ("mean", 2, 0, 0.9)],
)
def test_activation_visualization_output(
    Model, shape, layer_num, squeeze_op, interpolation, colormap, blend
):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.ActivationVisualization(model)

    output_image = interpretator(
        model.conv_layers[layer_num],
        input_image,
        squeeze_op=squeeze_op,
        interpolation=interpolation,
        colormap=colormap,
        blend=blend,
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape",
    [(inn.Inception5hModel, (200, 100, 3)), (inn.MNISTModel, (28, 28, 1))],
)
@pytest.mark.parametrize("blend", [-1.0, 1.01])
def test_activation_visualization_error(Model, shape, blend):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.ActivationVisualization(model)

    with pytest.raises(ValueError):
        interpretator(model.conv_layers, input_image, blend=blend)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, layer_num",
    [(inn.Inception5hModel, (500, 500, 3), 20), (inn.MNISTModel, (28, 28, 1), 1)],
)
def test_layer_activation_visualization(Model, shape, layer_num):
    model = Model()
    sess = model.start_session()

    layer_name = model.conv_layers_names[layer_num]
    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.LayerActivationVisualization(model)

    output_image = interpretator(layer_name, input_image)

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize("Model, shape, layer_num", [(inn.Inception5hModel, (100, 100, 3), 20)],)
@pytest.mark.parametrize("tiles", ["shift", "roll"])
@pytest.mark.parametrize("gradient_ascent", ["normal", "blurred", "smooth"])
def test_feature_visualization(Model, shape, layer_num, tiles, gradient_ascent):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.FeatureVisualization(model)

    output_image = interpretator(
        model.conv_layers[layer_num],
        input_image,
        num_epochs=1,
        num_octaves_per_epoch=2,
        tiles=tiles,
        gradient_ascent=gradient_ascent,
    )

    assert output_image.shape == input_image.shape
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize("Model, shape, layer_num", [(inn.MNISTModel, (28, 28, 1), 1)],)
@pytest.mark.parametrize("tiles", ["roll"])
def test_mnist_feature_visualization(Model, shape, layer_num, tiles):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.FeatureVisualization(model)

    output_image = interpretator(
        model.conv_layers[layer_num],
        input_image,
        num_epochs=1,
        num_octaves_per_epoch=0,
        tiles=tiles,
        tile_size=28,
    )

    assert output_image.shape == input_image.shape
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize("Model", [inn.Inception5hModel])
@pytest.mark.parametrize("layer_name", ["mixed4e_3x3_pre_relu/conv", "conv2d2_pre_relu/conv"])
def test_layer_visualization(Model, layer_name):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=(100, 100, 3)).astype(np.uint8)
    interpretator = inn.LayerVisualization(model)

    output_image = interpretator(
        layer_name, input_image, num_epochs=1, num_octaves_per_epoch=2,
    )

    assert output_image.shape == input_image.shape
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize("Model", [inn.Inception5hModel])
@pytest.mark.parametrize("layer_name", ["mixed4e_3x3_pre_relu/conv", "conv2d2_pre_relu/conv"])
@pytest.mark.parametrize("neuron_num", [1, 50])
def test_neuron_visualization(Model, layer_name, neuron_num):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=(100, 100, 3)).astype(np.uint8)
    interpretator = inn.NeuronVisualization(model)

    output_image = interpretator(
        layer_name, neuron_num, input_image, num_epochs=1, num_octaves_per_epoch=2,
    )

    assert output_image.shape == input_image.shape
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize("Model", [inn.Inception5hModel])
@pytest.mark.parametrize("class_num", [1, 50])
def test_output_class_visualization(Model, class_num):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=(100, 100, 3)).astype(np.uint8)
    interpretator = inn.OutputClassVisualization(model)

    output_image = interpretator(
        class_num, input_image, num_epochs=1, num_octaves_per_epoch=2,
    )

    assert output_image.shape == input_image.shape
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, tile_size",
    [(inn.Inception5hModel, (200, 200, 3), 20), (inn.MNISTModel, (28, 28, 1), 10)],
)
@pytest.mark.parametrize("pred", [3, None])
def test_occlusion(Model, shape, tile_size, pred):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.Occlusion(model)

    output_image = interpretator(
        input_image=input_image, pred=pred, tile_size=tile_size
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape",
    [(inn.Inception5hModel, (500, 500, 3)), (inn.MNISTModel, (28, 28, 1))],
)
@pytest.mark.parametrize("bg_threshold, fg_threshold", [(0.4, 0.6)])
def test_segmentation_mask(Model, shape, bg_threshold, fg_threshold):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    saliency_map = np.random.uniform(-1, 1, size=shape[0:2]).astype(np.uint8)
    interpretator = inn.SegmentationMask(model)

    output_image = interpretator(
        saliency_map=saliency_map,
        input_image=input_image,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, layer_num",
    [(inn.Inception5hModel, (200, 200, 3), 40), (inn.MNISTModel, (28, 28, 1), 1)],
)
@pytest.mark.parametrize("num_classes", [5, 1])
@pytest.mark.parametrize("guided_backpropagation", [True, False])
def test_grad_cam(Model, shape, layer_num, num_classes, guided_backpropagation):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.GradCAM(model)

    output_image = interpretator(
        xs_tensor=model.conv_layers[layer_num],
        input_image=input_image,
        num_classes=num_classes,
        guided_backpropagation=guided_backpropagation,
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, layer_num",
    [(inn.Inception5hModel, (200, 200, 3), 40), (inn.MNISTModel, (28, 28, 1), 1)],
)
def test_gradient_times_input(Model, shape, layer_num):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.GradientTimesInput(model)

    output_image = interpretator(
        loss_tensor=model.conv_layers[layer_num], input_image=input_image
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()


@pytest.mark.parametrize(
    "Model, shape, xs_layer_num, loss_layer_num",
    [
        (inn.Inception5hModel, (200, 200, 3), 20, 40),
        (inn.MNISTModel, (28, 28, 1), 0, 1),
    ],
)
def test_saliency_map(Model, shape, xs_layer_num, loss_layer_num):
    model = Model()
    sess = model.start_session()

    input_image = np.random.uniform(0, 255, size=shape).astype(np.uint8)
    interpretator = inn.SaliencyMap(model)

    output_image = interpretator(
        xs_tensor=model.conv_layers[xs_layer_num],
        loss_tensor=model.conv_layers[loss_layer_num],
        input_image=input_image,
    )

    assert output_image.shape[0:2] == input_image.shape[0:2]
    assert not np.array_equal(output_image, input_image)
    sess.close()
