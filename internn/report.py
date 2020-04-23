"""
Class for reporting parameters and results of computations.
"""
import numpy as np

class Reporter:
    
    def __init__(self, display=False):
        self.display = display


    def report_parameters(
            self, num_epochs, num_octaves_per_epoch, steps_per_octave,
            step_size, tile_size, tiles, gradient_ascent, grad_sigma, norm, op, octave_scale,
            blend, ksize, sigma, interpolation   
        ):
        """
        Prints parameters names with values.
        """
        if self.display:
            print("num_epochs={}".format(num_epochs))
            print("num_octaves_per_epoch={}".format(num_octaves_per_epoch))
            print("steps_per_octave={}".format(steps_per_octave))
            print("step_size={}".format(step_size))
            print("tile_size={}".format(tile_size))
            print("tiles={}".format(tiles))
            print("gradient_ascent={}".format(gradient_ascent))
            print("grad_sigma={}".format(grad_sigma))
            print("norm={}".format(norm))
            print("op={}".format(op))
            print("octave_scale={}".format(octave_scale))
            print("blend={}".format(blend))
            print("ksize={}".format(ksize))
            print("sigma={}".format(sigma))     
            print("interpolation={}".format(interpolation))   
        
        
    def report_feature_visualization(self, visualization_name):
        """
        Prints type of feature visualiation i.e. class name.
        """
        if self.display:
            print("visualization_name={}".format(visualization_name))
           
            
    def report_layer_visualization(self, layer_name):
        """
        Prints parameters specific to LayerVisualization. 
        """
        if self.display:
            print("layer_name={}".format(layer_name))
            
            
    def report_neuron_visualization(self, layer_name, neuron_num):
        """
        Prints parameters specific to NeuronVisualization. 
        """
        if self.display:
            print("layer_name={}".format(layer_name))
            print("neuron_num={}".format(neuron_num))

            
    def report_output_class_visualization(self, class_num):
        """
        Prints parameters specific to OutputClassVisualization. 
        """
        if self.display:
            print("class_num={}".format(class_num))
            
        
    def report_octave(self, losses, epoch, octave):
        """
        Prints epoch and octave numbers, loss mean, min and max.
        """
        if self.display: 
            msg = "Epoch: {}. Octave: {}. Loss mean {:.4f}, min: {:.4f}, max: {:.4f}."
            print(msg.format(epoch, octave, np.mean(losses), np.min(losses), np.max(losses)))
            
    def report_saliency_map(self, attribution_name, norm, norm_op, map_op):
        if self.display:
            print("attribution={}".format(attribution_name))
            print("norm={}".format(norm))
            print("norm_op={}".format(norm_op))
            print("map_op={}".format(norm_op))
