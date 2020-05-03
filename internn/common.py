"""
Methods commonly used in various classes.
"""
import cv2
import numpy as np
import tensorflow as tf


def activation_loss(activation_tensor, norm=2, op="mean"):
    """
    Returns norm of neuron activations in given tensor.

    PARAMETERS
    ----------
    activation_tensor : Tensor
        Tensor that contains neurons which norm is calculated.
    norm : int, optional
        Positive integer. Norm of neuron. The default is 2 which is euclidean norm.
    op : str, optional
        Operation which combines norms of neurons into one value. Acceptable values are: "mean",
        "max", "min", "std". The default is "mean".
            
    RETURNS
    -------
    float
        Norm of neuron activations.
    
    """
    norm = tf.constant([norm], dtype=activation_tensor.dtype)
    activation_tensor = tf.math.abs(activation_tensor)
    activation_tensor = tf.math.pow(x=activation_tensor, y=norm)

    if op == "mean":
        return tf.math.reduce_mean(input_tensor=activation_tensor)
    elif op == "max":
        return tf.math.reduce_max(input_tensor=activation_tensor)
    elif op == "min":
        return tf.math.reduce_max(input_tensor=activation_tensor)
    elif op == "std":
        return tf.math.reduce_std(input_tensor=activation_tensor)

    raise ValueError("Unsupported opperation: {}".format(op))


def create_gradient(
    xs_tensor, loss_tensor, loss_norm=2, loss_op="mean", guided_backpropagation=False,
):
    """
    Creates Tensor with gradients and loss.

    Parameters
    ----------
    xs_tensor : Tensor
        Tensor with respect to which the gradient is calculated.
    loss_tensor : Tensor
        Tensor which activations determine loss.
    loss_norm : int, optional
        Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
    loss_op : str, optional
        Operation which combines norms of neurons into one value. Acceptable values are "mean",
    guided_backpropagation : bool, optional
        Whether to use guided backpropagation. The default is False.

    Returns
    -------
    gradient_func
        Tensor with gradients.
    loss_func : Tensor
        Tensor with loss.

    """
    loss_func = activation_loss(loss_tensor, norm=loss_norm, op=loss_op)
    gradient_func = tf.gradients(ys=loss_func, xs=xs_tensor)[0]

    if guided_backpropagation:
        gradient_guided = tf.cast(gradient_func > 0, dtype="float32")
        xs_guided = tf.cast(xs_tensor > 0, dtype="float32")
        gradient_func = gradient_func * gradient_guided * xs_guided

    return gradient_func[0]


def create_heatmap(
    array,
    input_image,
    interpolation=cv2.INTER_LANCZOS4,
    colormap=cv2.COLORMAP_JET,
    blend=0.5,
):
    """
    Creates heatmap and blends it with input_image.

    Parameters
    ----------
    array : ndarray
        Values that determine heatmap.
    input_image : ndarray
        Image with which array is blended.
    interpolation : cv2 interpolation type, optional
        Parameter used to resize array to input_image's size. The default is cv2.INTER_LANCZOS4.
    colormap : cv2 colorMap, optional
        Colormap for heatmap. The default is COLORMAP_JET.    
    blend : float between 0.0 and 1.0, optional
        Blend factor for combining input_image with array. Setting blend to 1.0 would result in 
        returning not changed input_image.   

    Returns
    -------
    ndarray
        Created heatmap blended with input_image.

    """
    if array.shape != input_image.shape:
        height, width = input_image.shape[0:2]
        array = cv2.resize(array, dsize=(width, height), interpolation=interpolation)
    if input_image.shape[-1] == 1:
        input_image = input_image.copy()
        input_image = np.tile(input_image, 3)

    heatmap = cv2.applyColorMap(scale_to_image(array), colormap)
    heatmap = cv2.addWeighted(heatmap, 1 - blend, input_image, blend, 0)
    return heatmap


def squeeze_into_2D(array, op="max"):
    """
    Squeezes ndarray into 2D array.

    Parameters
    ----------
    array : ndarray
        Array to squeeze.
    op : str, optional
        Operation used to map values on axis into one value. Acceptable values are: "max", "min",
        "mean". The default is "max".

    Raises
    ------
    ValueError
        If op name is not acceptable.

    Returns
    -------
    ndarray
        Squeezed 2D array.

    """
    if len(array.shape) == 2:
        return array
    if op == "max":
        return array.max(axis=2)
    if op == "min":
        return array.min(axis=2)
    if op == "mean":
        return array.mean(axis=2)
    raise ValueError("Invalid op value: {}".format(op))


def scale_to_image(array):
    """
    Maps array values into [0, 255] and changes dtype to uint8.

    Parameters
    ----------
    array : ndarray
        Array to scale.

    Returns
    -------
    ndarray
        Scaled array.

    """
    array_scaled = (array - array.min()) / (array.max() - array.min())
    image = (255.0 * array_scaled).astype("uint8")
    return image
