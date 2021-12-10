from argparse import ArgumentParser
from typing import Dict


def parse_args(input_args) -> Dict[str, str]:
    parser = ArgumentParser()
    parser.add_argument("-m", "--movie-reviews", type=str, required=True)
    parser.add_argument("-o", "--operation", type=str, required=True)

    args = vars(parser.parse_args(input_args))
    return args
