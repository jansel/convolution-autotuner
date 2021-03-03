#!/usr/bin/env python
import sys

import torch

from convtuner import autotuner
from convtuner import testcases


def main():
    torch.set_num_threads(1)
    if sys.argv[1:] and sys.argv[1] == "autotune":
        autotuner.main(sys.argv[2:])
    else:
        testcases.main(sys.argv[1:])


if __name__ == "__main__":
    main()
