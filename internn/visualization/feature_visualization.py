"""
Implementations of feature visualization algotithms.
"""
import cv2
import numpy as np
import tensorflow as tf

import math

from internn.plot import Plotter
from internn.report import Reporter

def normalize_gradient(gradient):
    """
    Normalizes gradient based on its std.

    PARAMETERS
    ----------
    gradient : Tensor
        Initial gradient.

    RETURNS
    -------
    Tensor
        Normalizad gradient.
    """
    gradient = np.nan_to_num(gradient) / (np.nanstd(gradient) + 1e-8)
    return gradient


def gradient_ascent_normalized(image, gradient, step_size=2.0):
    """
    Perfoms standard gradinet ascent on image with normalization of gradient.

    PARAMETERS
    ----------
    image : ndarray
        Array that represents image.
    gradient : ndarray
        Array with calculated gradients.
    step_size : float, optional
        Size of gradient ascent step, similar to learning rate. The default is 2.0.

    RETURNS
    -------
    ndarray
        Updated image.
    """
    image += normalize_gradient(gradient) * step_size
    return image
    

def gradient_ascent_burred(image, gradient, step_size=2.0, grad_sigma=1.0):
    """
    Perfoms gradinet ascent on image with blurred, normalized gradient.

    PARAMETERS
    ----------
    image : ndarray
        Array that represents image.
    gradient : ndarray
        Array with calculated gradients.
    step_size : float, optional
        Size of gradient ascent step, similar to learning rate. The default is 2.0.
    grad_sigma : float, optional
        Sigma parameter for gaussian blur. The default is 1.0.

    RETURNS
    -------
    ndarray
        Updated image.
    """
    gradient_blurred = cv2.GaussianBlur(src=gradient, ksize=(1,1), sigmaX=grad_sigma)
    return gradient_ascent_normalized(image, gradient_blurred, step_size)


def gradient_ascent_smooth(image, gradient, step, step_size=2.0, grad_sigma=(0.15,0.5)):
    """
    Perfoms gradinet ascent on image with new gradient created by adding initial gradient 3 times,
    each time with differently parametrized gaussian blur. I also normalizes this gradient.

    PARAMETERS
    ----------
    image : ndarray
        Array that represents image.
    gradient : ndarray
        Array with calculated gradients.
    step : int
        Step number used to calculate gaussian blur sigma specifically for this step.
    step_size : float, optional
        Size of gradient ascent step, similar to learning rate. The default is 2.0.
    grad_sigma : typle with two floats, optional
        Parameters to linear function that determines gausian blur for specific gradient step.
        The default is (0.15, 0.5).

    RETURNS
    -------
    ndarray
        Updated image.
    """
    alpha, const, factors = grad_sigma
    sigma = alpha * step + const
    smooth_gradient = np.zeros_like(gradient)
    for factor in [0.5, 1.0, 2.0]:
        smooth_gradient += cv2.GaussianBlur(src=gradient, ksize=(1,1), sigmaX=sigma*factor)
        
    return gradient_ascent_normalized(image, smooth_gradient, step_size)
    

def random_roll(array):    
    """
    Rolls array randomly vertically and horizontally.

    PARAMETERS
    ----------
    image : ndarray
        Array.

    RETURNS
    -------
    image_rolled : ndarray
        Rolled array.
    height_roll : int
        Size of roll that was done vertically.
    width_roll : int
        Size of roll that was done horizontally.
    """
    height, width, _ = array.shape
    height_roll = np.random.random_integers(-height, height)
    width_roll = np.random.random_integers(-width, width)
    
    array_rolled = np.roll(array, height_roll, axis=0)    
    array_rolled = np.roll(array_rolled, width_roll, axis=1)
    
    return array_rolled, height_roll, width_roll


def unroll(array, height_shift, width_shift):
    """
    Unrolls previously done roll.

    PARAMETERS
    ----------
    array : ndarray
        Array.
    height_roll : int
        Size of roll that was done vertically.
    width_roll : int
        Size of roll that was done horizontally.

    RETURNS
    -------
    ndarray
        Unrolled array.
    """
    gradient_unrolled = np.roll(gradient_unrolled, -height_shift, axis=0)
    gradient_unrolled = np.roll(gradient, -width_shift, axis=1)
    
    return gradient_unrolled

    
def tile_real_size(size, tile_size=512):
    """
    Calculates optimal tile size, so that the last tile is not much smaller.

    PARAMETERS
    ----------
    size : ine
        Size of array that will be divided into tiles.
    tile_size : ine, optional
        Suggested size of tile. The default is 512.

    RETURNS
    -------
    int
        Optiomal tile size.
    """
    num_tiles = max(size / tile_size, 1)
    return math.ceil(size / num_tiles)


class FeatureVisualization:
    
    def __init__(self, model, reporter=None, plotter=None):
        self.model = model
        self.reporter = Reporter() if reporter is None else reporter
        self.plotter = Plotter() if plotter is None else plotter
      

    def __call__(
            self, feature_tensor, input_image, num_epochs=5, num_octaves_per_epoch=5,
            steps_per_octave=10, step_size=2.0, tile_size=512, tiles="shift",
            gradient_ascent="normal", grad_sigma=None, norm=2, op="mean", octave_scale=0.7,
            blend=0.2, ksize=(1,1), sigma=1, interpolation=cv2.INTER_LANCZOS4 
        ):
        """
        Performs feature visualization on image by maximazing activations in given activation
        tensor.

        PARAMETERS
        ----------
        feature_tensor : Tensor
            Tensor with activations which norm will be maximized.
        input_image : ndarray
            Image which will be upaded, so that activations for this image are maximized.
        num_epochs : int, optional
            Number of epochs. The default is 5.
        num_octaves_per_epoch : int, optional
            Number of octaves in one epoch. The default is 5.
        steps_per_octave : int, optional
            Number of gradient ascent steps in one octave. The default is 10.
        step_size : float, optional
            Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
        gradient_ascent : str, optional
            Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
            default is "normal".
        grad_sigma : TYPE, optional
            Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
            1 for "blurred" and (0.15, 0.5) for "smooth".
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std".
        octave_scale : float, optional
            Scaling factor in octave. The default is 0.7.
        blend : float, optional
            Factor for blending input and output images after octave step. The default is 0.2,
            which means that blended image will consist of 0.2 of input image and 0.8 of output
            image.
        ksize : tuple of two ints, optional
            Parameter for resizing image in octave. The default is (1,1).
        sigma : float, optional
            Parameter for gaussian blur performed on input image in octave. The default is 1.
        interpolation : TYPE, optional
            Parameter for gaussian blur performed on input image in octave. The default is 
            cv2.INTER_LANCZOS4.

        RETURNS
        -------
        ndarray
            Input image that maximizes activations in given tensor.
        """
        if self.__class__.__name__ == "FeatureVisualization":
            self.reporter.report_feature_visualization(self.__class__.__name__)
            
        self.reporter.report_parameters(
            num_epochs=num_epochs,
            num_octaves_per_epoch=num_octaves_per_epoch,
            steps_per_octave=steps_per_octave,
            step_size=step_size,
            tile_size=tile_size,
            tiles=tiles,
            gradient_ascent=gradient_ascent,
            grad_sigma=grad_sigma,
            norm=norm,
            op=op,
            octave_scale=octave_scale,
            blend=blend,
            ksize=ksize,
            sigma=sigma,
            interpolation=interpolation
            )   
                 
        image = input_image.copy().astype(np.float64)
        
        for epoch in range(num_epochs):
            image = self.epoch_step(
                feature_tensor=feature_tensor,
                input_image=image,
                epoch = epoch,
                num_octaves_per_epoch=num_octaves_per_epoch,
                steps_per_octave=steps_per_octave,
                step_size=step_size,
                tile_size=tile_size,
                tiles=tiles,
                gradient_ascent=gradient_ascent,
                grad_sigma=grad_sigma,
                norm=norm,
                op=op,
                octave_scale=octave_scale,
                blend=blend,
                ksize=ksize,
                sigma=sigma,
                interpolation=interpolation
                )

        return image
    
    
    def epoch_step(
            self, feature_tensor, input_image, epoch, num_octaves_per_epoch=5, steps_per_octave=10, 
            step_size=2.0, tile_size=512, tiles="shift", gradient_ascent="normal", grad_sigma=None, 
            norm=2, op="mean", octave_scale=0.7, blend=0.2, ksize=(1,1), sigma=1, 
            interpolation=cv2.INTER_LANCZOS4
            ):
        """
        Epoch step of feature visualization.

        PARAMETERS
        ----------
        feature_tensor : Tensor
            Tensor with activations which norm will be maximized.
        input_image : ndarray
            Image which will be upaded, so that activations for this image are maximized.
        epoch : int
            Epoch number.
        num_octaves_per_epoch : int, optional
            Number of octaves in one epoch. The default is 5.
        steps_per_octave : int, optional
            Number of gradient ascent steps in one octave. The default is 10.
        step_size : float, optional
            Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
        gradient_ascent : str, optional
            Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
            default is "normal".
        grad_sigma : float or tuple of two floats, optional
            Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
            1 for "blurred" and (0.15, 0.5) for "smooth".
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std".
        octave_scale : float, optional
            Scaling factor in octave. The default is 0.7.
        blend : float, optional
            Factor for blending input and output images after octave step. The default is 0.2,
            which means that blended image will consist of 0.2 of input image and 0.8 of output
            image.
        ksize : tuple of two ints, optional
            Parameter for resizing image in octave. The default is (1,1).
        sigma : float, optional
            Parameter for gaussian blur performed on input image in octave. The default is 1.
        interpolation : TYPE, optional
            Parameter for gaussian blur performed on input image in octave. The default is 
            cv2.INTER_LANCZOS4.

        RETURNS
        -------
        ndarray
            Input image that maximizes activations in given tensor.
        """    
        
        if num_octaves_per_epoch > 0:
            
            image_blurred = cv2.GaussianBlur(src=input_image, ksize=ksize, sigmaX=sigma)

            new_height = int(image_blurred.shape[0] * octave_scale)
            new_width = int(image_blurred.shape[1] * octave_scale)
            dsize = (new_width, new_height)
            scaled_input_image = cv2.resize(image_blurred, dsize=dsize, interpolation=interpolation)
            
            output_image = self.epoch_step(
                feature_tensor=feature_tensor,
                input_image=scaled_input_image,
                epoch = epoch,
                num_octaves_per_epoch=num_octaves_per_epoch - 1,
                steps_per_octave=steps_per_octave,
                step_size=step_size,
                tile_size=tile_size,
                tiles=tiles,
                gradient_ascent=gradient_ascent,
                grad_sigma=grad_sigma,
                norm=norm, op=op,
                octave_scale=octave_scale,
                blend=blend,
                ksize=ksize,
                sigma=sigma,interpolation=interpolation
                )

            height, width = input_image.shape[0:2]
            dsize = (width, height)
            output_image = cv2.resize(output_image, dsize=dsize, interpolation=interpolation)
            input_image = cv2.addWeighted(input_image, blend, output_image, 1-blend, 0)

        output_image = self.octave_step(
               feature_tensor=feature_tensor,
               input_image=input_image,
               epoch = epoch,
               octave = num_octaves_per_epoch,
               steps_per_octave=steps_per_octave,
               step_size=step_size,
               tile_size=tile_size,
               norm=norm, op=op,
               grad_sigma=grad_sigma,
               gradient_ascent=gradient_ascent,
               tiles=tiles
               )
            
        return output_image
        
        
    def octave_step(
            self, feature_tensor, input_image, epoch, octave, steps_per_octave=10, step_size=2.0, 
            tile_size=512, tiles="shift", gradient_ascent="normal",  grad_sigma=None, norm=2, 
            op="mean"     
        ):
        """
        Octave step of feature visualization.   

        PARAMETERS
        ----------
        feature_tensor : Tensor
            Tensor with activations which norm will be maximized.
        input_image : ndarray
            Image which will be upaded, so that activations for this image are maximized.
        epoch : int
            Epoch number.
        octave : int
            Octave number.
        steps_per_octave : int, optional
            Number of gradient ascent steps in one octave. The default is 10.
        step_size : float, optional
            Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
        gradient_ascent : str, optional
            Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
            default is "normal".
        grad_sigma : float or tuple of two floats, optional
            Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
            1 for "blurred" and (0.15, 0.5) for "smooth".
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std".

        RAISES
        ------
        ValueError
            Unsupported gradient_ascent value.

        RETURNS
        -------
        ndarray
            Input image that maximizes activations in given tensor.
        """
        image = input_image.copy().astype(np.float64)
        loss_func = self.model.loss(feature_tensor, norm=norm, op=op)
        losses = list()

        for step in range(steps_per_octave):

            loss_result, gradient_result = self.tiled_gradient(
                loss_func=loss_func, 
                image=image, 
                tile_size=tile_size,
                tiles=tiles
                )

            losses.append(loss_result)
            
            if gradient_ascent == "normal":
                image = gradient_ascent_normalized(image, gradient_result, step_size)
            elif gradient_ascent == "blurred":
                grad_sigma = 1.0 if grad_sigma is None else grad_sigma
                image = gradient_ascent_blurred(image, gradient_result, step_size, grad_sigma)
            elif gradient_ascent == "smooth":
                grad_sigma = (4.0,0.5,[0.5,1.0,2.0]) if grad_sigma is None else grad_sigma
                image = gradient_ascent_smooth(image, gradient_result, step, step_size, grad_sigma)
            else:
                raise ValueError("Unsupported gradient_ascent value {}.".format(gradient_ascent))
            
        self.reporter.report_octave(losses, epoch, octave)
        self.plotter.plot_losses(losses,epoch, octave)
        self.plotter.plot_image(image, epoch, octave)
            
        return image
  

    def tiled_gradient(self, loss_func, image, tile_size=512, tiles="shift"):
        """
        Divides image into smaller tiles and performs gradient on each of them.

        PARAMETERS
        ----------
        loss_func : Operation
            Operations that calculates loss.
        image : TYPE
            Input image for loss function.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".

        RAISES
        ------
        ValueError
            Unsupported tiles value.

        RETURNS
        -------
        loss : float
            Sum of losses in tiles.
        gradient : ndarray
            Calculated gradient.
        """
        gradient_func = tf.gradients(ys=loss_func, xs=self.model.input)[0]
        gradient_func /= (tf.math.reduce_std(gradient_func) + 1e-8)
        
        gradient = np.zeros_like(image)
        loss = 0

        height, weight, _ = image.shape
        tile_height = tile_real_size(height, tile_size)
        tile_weight = tile_real_size(weight, tile_size)
        
        if tiles == "roll":
            image, height_roll, width_roll = random_roll(image)
        elif tiles == "shift":
            start_height = int(np.random.uniform(-0.75, -0.25) * tile_height)
            start_weight = int(np.random.uniform(-0.75, -0.25) * tile_weight) 
        else: raise ValueError("Unsupported tiles value: {}.".format(tiles))
            
        hs = range(start_height, height, tile_height)
        ws = range(start_weight, weight, tile_weight)
        
        for h in hs:      
            for w in ws:
                
                image_tile = image[max(h, 0):h + tile_height, max(w, 0):w+tile_weight, :]         
                image_tile = np.expand_dims(image_tile, axis=0)

                sess = tf.compat.v1.get_default_session()
                
                try:
                    feed_dict = {self.model.input_name : image_tile}
                    tile_loss, tile_gradient = sess.run([loss_func, gradient_func], feed_dict)
                except tf.errors.InvalidArgumentError as e:
                    print("Tile skiped: {}".format(e.message))
                    continue
                    
                loss += np.nan_to_num(tile_loss)
                gradient[max(h, 0):h + tile_height, max(w, 0):w + tile_weight, :] = tile_gradient
            
        if tiles == "roll":
            gradient = unroll(grad, height_roll, width_roll)
            
        return loss, gradient

    
class LayerVisualization(FeatureVisualization):
    
    def __call__(
            self, layer_name, input_image, num_epochs=5, num_octaves_per_epoch=5,
            steps_per_octave=10, step_size=2.0, tile_size=512, tiles="shift",
            gradient_ascent="normal", grad_sigma = None, norm=2, op="mean", octave_scale=0.7,
            blend=0.2, ksize=(1,1), sigma=1, interpolation=cv2.INTER_LANCZOS4
        ):
         
        """
        Calls FeatureVisualization for given layer.
        
        PARAMETERS
        ----------
        layer_name : str
            Name of layer.
        input_image : ndarray
            Image which will be upaded, so that activations for this image are maximized.
        num_epochs : int, optional
            Number of epochs. The default is 5.
        num_octaves_per_epoch : int, optional
            Number of octaves in one epoch. The default is 5.
        steps_per_octave : int, optional
            Number of gradient ascent steps in one octave. The default is 10.
        step_size : float, optional
            Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
        gradient_ascent : str, optional
            Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
            default is "normal".
        grad_sigma : TYPE, optional
            Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
            1 for "blurred" and (0.15, 0.5) for "smooth".
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std".
        octave_scale : float, optional
            Scaling factor in octave. The default is 0.7.
        blend : float, optional
            Factor for blending input and output images after octave step. The default is 0.2,
            which means that blended image will consist of 0.2 of input image and 0.8 of output
            image.
        ksize : tuple of two ints, optional
            Parameter for resizing image in octave. The default is (1,1).
        sigma : float, optional
            Parameter for gaussian blur performed on input image in octave. The default is 1.
        interpolation : TYPE, optional
            Parameter for gaussian blur performed on input image in octave. The default is 
            cv2.INTER_LANCZOS4.

        RETURNS
        -------
        ndarray
            Input image that maximizes activations in given tensor.
        """
        if self.__class__.__name__ == "LayerVisualization":
            self.reporter.report_feature_visualization(self.__class__.__name__)
            self.reporter.report_layer_visualization(layer_name)

        layer_tensor =  self.model.find_layer_tensor(layer_name)
        return super().__call__(
            feature_tensor=layer_tensor,
            input_image=input_image,
            num_epochs=num_epochs,
            num_octaves_per_epoch=num_octaves_per_epoch,
            steps_per_octave=steps_per_octave,
            step_size=step_size,
            tile_size=tile_size,
            tiles=tiles,
            gradient_ascent=gradient_ascent,
            grad_sigma=grad_sigma,
            norm=norm,
            op=op,
            octave_scale=octave_scale,
            blend=blend,
            ksize=ksize,
            sigma=sigma,
            interpolation=interpolation
            )
    
                        
class NeuronVisualization(FeatureVisualization):

    def __call__(
            self, layer_name, neuron_num, input_image, num_epochs=5, num_octaves_per_epoch=5,
            steps_per_octave=10, step_size=2.0, tile_size=512, tiles="shift",
            gradient_ascent="normal", grad_sigma = None, norm=2, op="mean", octave_scale=0.7,
            blend=0.2, ksize=(1,1), sigma=1, interpolation=cv2.INTER_LANCZOS4 
        ):
        """
        Calls FeatureVisualization for specific neuron.
        
        PARAMETERS
        ----------
        layer_name : str
            Name of layer.
        neuron_num : int
            Neuron number in layer.
        input_image : ndarray
            Image which will be upaded, so that activations for this image are maximized.
        num_epochs : int, optional
            Number of epochs. The default is 5.
        num_octaves_per_epoch : int, optional
            Number of octaves in one epoch. The default is 5.
        steps_per_octave : int, optional
            Number of gradient ascent steps in one octave. The default is 10.
        step_size : float, optional
            Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
        tile_size : int, optional
            Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
            so that the last tile is not much smaller. The default is 512.
        tiles : str, optional
            Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
        gradient_ascent : str, optional
            Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
            default is "normal".
        grad_sigma : TYPE, optional
            Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
            1 for "blurred" and (0.15, 0.5) for "smooth".
        norm : int, optional
            Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
        op : str, optional
            Operation which combines norms of neurons into one value. Acceptable values are "mean",
           "max", "min", "std".
        octave_scale : float, optional
            Scaling factor in octave. The default is 0.7.
        blend : float, optional
            Factor for blending input and output images after octave step. The default is 0.2,
            which means that blended image will consist of 0.2 of input image and 0.8 of output
            image.
        ksize : tuple of two ints, optional
            Parameter for resizing image in octave. The default is (1,1).
        sigma : float, optional
            Parameter for gaussian blur performed on input image in octave. The default is 1.
        interpolation : TYPE, optional
            Parameter for gaussian blur performed on input image in octave. The default is 
            cv2.INTER_LANCZOS4.

        RETURNS
         -------
         ndarray
             Input image that maximizes activations in given tensor.
         """
        if self.__class__.__name__ == "NeuronVisualization":
            self.reporter.report_feature_visualization(self.__class__.__name__)
            self.reporter.report_neuron_visualization(layer_name, neuron_num)
            
        neuron_tensor = self.model.find_neuron_tensor(layer_name, neuron_num)
        
        return super().__call__(
             feature_tensor=neuron_tensor,
             input_image=input_image,
             num_epochs=num_epochs,
             num_octaves_per_epoch=num_octaves_per_epoch,
             steps_per_octave=steps_per_octave,
             step_size=step_size,
             tile_size=tile_size,
             tiles=tiles,
             gradient_ascent=gradient_ascent,
             grad_sigma=grad_sigma,
             norm=norm,
             op=op,
             octave_scale=octave_scale,
             blend=blend,
             ksize=ksize,
             sigma=sigma,
             interpolation=interpolation
             )

    
class OutputClassVisualization(NeuronVisualization):
    
     def __call__(
            self, class_num, input_image, num_epochs=5, num_octaves_per_epoch=5,
            steps_per_octave=10, step_size=2.0, tile_size=512, tiles="shift",
            gradient_ascent="normal", grad_sigma = None, norm=2, op="mean", octave_scale=0.7,
            blend=0.2, ksize=(1,1), sigma=1, interpolation=cv2.INTER_LANCZOS4
        ):
         """
         Calls FeatureVisuzalization for certain class by calling it on neuron in last the layer.

         PARAMETERS
         ----------
         class_num : int or None
             Class number. If None, visualization is done for predicted class.
         input_image : ndarray
             Image which will be upaded, so that activations for this image are maximized.
         num_epochs : int, optional
             Number of epochs. The default is 5.
         num_octaves_per_epoch : int, optional
             Number of octaves in one epoch. The default is 5.
         steps_per_octave : int, optional
             Number of gradient ascent steps in one octave. The default is 10.
         step_size : float, optional
             Size of step in gradient ascent. Similar to learning rate. The default is 2.0.
         tile_size : int, optional
             Size of tile, so that bigger images are splitted into tiles. Tile size can be modified,
             so that the last tile is not much smaller. The default is 512.
         tiles : str, optional
             Way of obtaining tiles. Accepted values are: "shift" and "roll". The default is "shift".
         gradient_ascent : str, optional
             Gradient ascent version. Accepted values are: "normal", "blurred", "smooth". The 
             default is "normal".
         grad_sigma : TYPE, optional
             Sigma parameter in gaussian blur performed on calculated gradient. If None, it is equal
             1 for "blurred" and (0.15, 0.5) for "smooth".
         norm : int, optional
             Positve integer. Norm of neuron. The default is 2 which is euclidean norm.
         op : str, optional
             Operation which combines norms of neurons into one value. Acceptable values are "mean",
            "max", "min", "std".
         octave_scale : float, optional
             Scaling factor in octave. The default is 0.7.
         blend : float, optional
             Factor for blending input and output images after octave step. The default is 0.2,
             which means that blended image will consist of 0.2 of input image and 0.8 of output
             image.
         ksize : tuple of two ints, optional
             Parameter for resizing image in octave. The default is (1,1).
         sigma : float, optional
             Parameter for gaussian blur performed on input image in octave. The default is 1.
         interpolation : TYPE, optional
             Parameter for gaussian blur performed on input image in octave. The default is 
             cv2.INTER_LANCZOS4.

         RETURNS
         -------
         ndarray
             Input image that maximizes activations in given tensor.
         """
         if self.__class__.__name__ == "OutputClassVisualization":
             self.reporter.report_feature_visualization(self.__class__.__name__)
             self.reporter.report_output_class_visualization(class_num)
             
         if class_num is None:
             input_image_expanded = np.expand_dims(input_image, axis=0)
             feed_dict = {self.model.input_name: input_image_expanded}
             sess = tf.compat.v1.get_default_session()
             output = sess.run(self.model.output, feed_dict)[0]
             class_num = np.argmax(output)
         
         return super().__call__(
             layer_name=self.model.output_name,
             neuron_num=class_num,
             input_image=input_image,
             num_epochs=num_epochs,
             num_octaves_per_epoch=num_octaves_per_epoch,
             steps_per_octave=steps_per_octave,
             step_size=step_size,
             tile_size=tile_size,
             tiles=tiles,
             gradient_ascent=gradient_ascent,
             grad_sigma=grad_sigma,
             norm=norm,
             op=op,
             octave_scale=octave_scale,
             blend=blend,
             ksize=ksize,
             sigma=sigma,
             interpolation=interpolation
             )
