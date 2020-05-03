"""
Creating, displaying and saving plots, images.
"""
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from os import path
from pathlib import Path
from PIL import Image



def create_filename(filename):
    """
    Generates filename from filename and date.

    PARAMETERS
    ----------
    filename : str
        Name of file.

    RETURNS
    -------
    str
        Name filename.
    """
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    return "{}_{}.png".format(now_str, filename)


class Plotter:
    """
    Class for displaying and saving images.
    """

    def __init__(self, save_dir="", display=False, save=False):
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display = display
        self.save = save

    def plot_image(self, image, filename):
        """
        Displays and saves array as image. Prints path to saved file.
        
        PARAMETERS
        ----------
        image : array-like
            Image to display, save.
        filename_suffix : str, optional
            Suffix to add in filename e.g. with epoch number. The default is "".
        """
        new_filename = create_filename(filename)

        if self.display:
            plt.axis("off")
            plt.imshow(Image.fromarray(image))
            plt.show()

        if self.save:
            file_path = path.join(self.save_dir, new_filename)
            cv2.imwrite(file_path, image)
            print("Image saved: {}.".format(file_path))

        plt.close()

    def plot_losses(self, losses, title_suffix="", filename_suffix=""):
        """
        Displays and saves losses plot. Prints path to saved file.
        
        PARAMETERS
        ----------
        losses : list of numbers
            Losses to plot.
        title_suffix : str, optional
            Suffix to add in plot title e.g. with algorithm name, epoch number. The default is "".
        filename_suffix : str, optional
            Suffix to add in filename e.g. with algorithm name, epoch number. The default is "".
        """

        if self.display:
            plt.plot(losses)
            plt.title("Losses\n{}".format(title_suffix))
            plt.show()

        if self.save:
            plt.plot(losses)
            plt.title("Losses\n{}".format(title_suffix))
            filename = create_filename("losses{}".format(filename_suffix))
            file_path = path.join(self.save_dir, filename)
            plt.savefig(file_path)
            print("Losses plot saved: {}.".format(file_path))

        plt.close()
