import pytest

import numpy as np
import shutil

from datetime import datetime

import internn as inn


@pytest.mark.parametrize(
    "display", [True, False],
)
@pytest.mark.parametrize(
    "save", [True, False],
)
def test_plot_image(display, save, capsys):
    image = np.random.uniform(0, 255, size=(100, 100, 3)).astype(np.uint8)

    save_dir = str(datetime.now())
    plotter = inn.Plotter(save_dir=save_dir, display=display, save=save)
    plotter.plot_image(image, filename="test_name")

    captured = capsys.readouterr()

    if display:
        assert "Figure" in captured.out
    else:
        assert "Figure" not in captured.out
    if save:
        assert "Image saved: {}".format(save_dir) in captured.out
        assert "test_name" in captured.out
    else:
        assert "Image saved: {}".format(save_dir) not in captured.out
        assert "test_name" not in captured.out

    shutil.rmtree(save_dir)


@pytest.mark.parametrize(
    "display, save", [(True, True), (True, False), (False, True), (False, False)]
)
def test_plot_losses(display, save, capsys):
    losses = [7, 10, 4]

    save_dir = str(datetime.now())
    plotter = inn.Plotter(save_dir=save_dir, display=display, save=save)
    plotter.plot_losses(losses, title_suffix="tile_suf", filename_suffix="_suf")

    captured = capsys.readouterr()

    if display:
        assert "Figure" in captured.out
    else:
        assert "Figure" not in captured.out
    if save:
        assert "Losses plot saved: ".format(save_dir) in captured.out
        assert "losses_suf" in captured.out
    else:
        assert "Losses plot saved: {}".format(save_dir) not in captured.out
        assert "losses_suf" not in captured.out

    shutil.rmtree(save_dir)
