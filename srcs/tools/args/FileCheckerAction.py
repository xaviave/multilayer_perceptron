import os
import argparse


class FileCheckerAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values) or os.path.splitext(values)[1] != ".csv":
            raise ValueError(
                f"File '{values}' does not exist or is in the wrong format (CSV)"
            )
        setattr(namespace, self.dest, values)
