#!/usr/bin/env python
import argparse
import ast
import json
import logging
import multiprocessing
import os
import timeit
from unittest.mock import patch

import opentuner
import torch
import threading
from numpy import median
from opentuner import Result
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator

from codegen import ConvolutionGenerator
from config_recorder import ConfigProxy, ConfigRecorder
from evaluate import gflops

log = logging.getLogger(__name__)
lock = threading.Lock()
REPEAT = 1

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("--conv2d")
parser.add_argument("--input")
parser.set_defaults(**vars(parser.parse_args([
    "--no-dups",
    # "--stop-after=600",
    f"--parallelism={multiprocessing.cpu_count()}",
    "--parallel-compile",
    "--technique=AUCBanditMetaTechniqueB",
    "--conv2d", "(576, 1280, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros')",
    "--input", "(1, 576, 1, 1)",
])))


class ConvTuner(MeasurementInterface):
    def __init__(self, args=None):
        manipulator = ConfigurationManipulator()
        conv_args = ast.literal_eval(args.conv2d)
        input_shape = ast.literal_eval(args.input)
        self.conv = torch.nn.Conv2d(*conv_args)
        self.image = torch.randn(*input_shape)

        cg = ConvolutionGenerator(ConfigRecorder(manipulator), self.conv, self.image, subproc=False)
        assert len(manipulator.params)
        sec = float(median(timeit.repeat(lambda: cg(self.image), number=1, repeat=10)))
        self.times = max(1, int(0.1 / sec))

        super(ConvTuner, self).__init__(
            args=args,
            project_name="ConvTuner",
            program_name=repr(conv_args),
            program_version=repr(input_shape),
            manipulator=manipulator,
        )

    def compile_and_run(self, desired_result, input, limit):
        return Result(time=self.measure_cfg(desired_result.configuration.data))

    def measure_cfg(self, cfg):
        cg = ConvolutionGenerator(ConfigProxy(cfg), self.conv, self.image)
        cg(self.image)  # warmup
        sec = median(timeit.repeat(lambda: cg(self.image), number=self.times, repeat=REPEAT))
        cg.close()
        return float(sec)

    def compile(self, cfg, id):
        return ConvolutionGenerator(ConfigProxy(cfg), self.conv, self.image, subproc=True)

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        with lock:
            try:
                compile_result(self.image)  # warmup
                sec = median(timeit.repeat(lambda: compile_result(self.image), number=self.times, repeat=REPEAT))
                return Result(time=float(sec))
            finally:
                compile_result.close()

    def save_final_config(self, configuration):
        cfg = configuration.data
        sec = self.measure_cfg(cfg)
        gf = gflops(self.conv, self.image) * self.times / sec
        print(f"Final configuration ({sec:.2f}s, {gf:.1f} gflops): {self.config_filename()}")
        with open(self.config_filename(), "w") as fd:
            json.dump(cfg, fd, indent=2)
            fd.write("\n")

    def config_filename(self):
        os.path.exists("configs") or os.mkdir("configs")
        return f"./configs/{self.program_name()},{self.program_version()}.json"


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    opentuner.init_logging()
    ConvTuner.main(parser.parse_args())
