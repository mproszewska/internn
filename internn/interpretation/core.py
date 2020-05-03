"""
Base for every interpretation class.
"""
from internn.plot import Plotter
from internn.report import Reporter


class Interpretation:
    """
    Base class that each class with interpretation algorithm extends.
    """

    def __init__(self, model, reporter=None, plotter=None):
        self.model = model
        self.reporter = Reporter() if reporter is None else reporter
        self.plotter = Plotter() if plotter is None else plotter

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
