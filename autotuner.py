#!/usr/bin/env python
import argparse
import ast
import json
import logging
import os
import timeit
from unittest.mock import patch

import opentuner
import torch
from numpy import median
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator

import codegen
from codegen import ConvolutionGenerator
from config_recorder import ConfigProxy, ConfigRecorder
from evaluate import gflops

log = logging.getLogger(__name__)
REPEAT = 3

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("--conv2d")
parser.add_argument("--input")
parser.set_defaults(**vars(parser.parse_args([
    "--no-dups",
    # "--stop-after=600",
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

        with patch.object(codegen, "cfg", ConfigRecorder(manipulator)):
            cg = ConvolutionGenerator(self.conv, self.image)
            log.info("{len(manipulator.params)} params")
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

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        return opentuner.resultsdb.models.Result(time=self.measure_cfg(cfg))

    def measure_cfg(self, cfg):
        with patch.object(codegen, "cfg", ConfigProxy(cfg)):
            cg = ConvolutionGenerator(self.conv, self.image)
            cg(self.image)  # warmup
            sec = median(timeit.repeat(lambda: cg(self.image), number=self.times, repeat=REPEAT))
            cg.close()
            return float(sec)

    def save_final_config(self, configuration):
        """
        called at the end of autotuning with the best resultsdb.models.Configuration
        """
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
    ConvTuner.main(parser.parse_args())
