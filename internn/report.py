"""
Reporting parameters and results of computations.
"""


class Reporter:
    """
    Class for reporting parameters and results of computations.
    """

    def __init__(self, display=False):
        self.display = display

    def report_parameters(self, params):
        """
        Prints parameters names with values.
        """
        if self.display:
            for key, value in params.items():
                print("{}={}".format(key, value))

    def report_message(self, msg):
        """
        Prints message.
        """
        if self.display:
            print(msg, end="")
