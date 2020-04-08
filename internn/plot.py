"""
Class for creating, displaying and saving plots, images.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from os import path
from pathlib import Path

def create_filename(plot_name, epoch, octave):
    """
    Generates filename from date, epoch, octave and name of plot i.e. image, losses.

    PARAMETERS
    ----------
    plot_name : str
        Plot name, one of: gradient, image.
    epoch : int
        Epoch number.
    octave : int
        Octave number.

    RETURNS
    -------
    str
        Name for plot.
    """
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    return "{}_epoch-{}_octave-{}_{}.png".format(now_str, epoch, octave, plot_name)
    

class Plotter:
    
    def __init__(self, save_dir="", display=False, save=False):
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display = display
        self.save= save
            
               
    def plot_image(self, image, epoch, octave):
        """
        Displays and saves array as image. Prints path to saved file.
        
        PARAMETERS
        ----------
        image : array-like
            Image to display, save.
        epoch : int
            Epoch number included in name of saved file.
        octave : int
            Octave number included in name of saved file.
        """
        filename = create_filename("image", epoch, octave)
        
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
        
        if self.display:
            cv2.imshow(filename, image)
            
        if self.save:         
            file_path = path.join(self.save_dir, filename)
            cv2.imwrite(file_path, image)
            print("Image saved: {}.".format(file_path))
            
        plt.close()
            
        
    def plot_losses(self, losses, epoch, octave):
        """
        Displays and saves losses plot. Prints path to saved file.
        
        PARAMETERS
        ----------
        losses : list of numbers
            Losses to plot.
        epoch : int
            Epoch number included in name of saved file.
        octave : int
            Octave number included in name of saved file.
        """
        plt.plot(losses)
        plt.title("Losses\nepoch: {} octave: {}".format(epoch, octave))
        
        if self.display:
            plt.show()
            
        if self.save:
            filename = create_filename("losses", epoch, octave)
            file_path = path.join(self.save_dir, filename)
            plt.savefig(file_path)
            print("Losses plot saved: {}.".format(file_path))
            
        plt.close()