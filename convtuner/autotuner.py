import argparse
import ast
import json
import logging
import multiprocessing
import os
import timeit

import opentuner
import torch
from numpy import median
from opentuner import Result
from opentuner.measurement import MeasurementInterface
from opentuner.search.bandittechniques import AUCBanditMetaTechnique
from opentuner.search.evolutionarytechniques import NormalGreedyMutation, UniformGreedyMutation
from opentuner.search.manipulator import ConfigurationManipulator, NumericParameter, BooleanParameter, \
    PermutationParameter

from convtuner.codegen import ConvolutionGenerator
from convtuner.config_recorder import ConfigProxy, ConfigRecorder
from convtuner.utils import gflops

log = logging.getLogger(__name__)
REPEAT = 1

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("--conv2d")
parser.add_argument("--input")
parser.set_defaults(**vars(parser.parse_args([
    "--no-dups",
    # "--stop-after=600",
    f"--parallelism={multiprocessing.cpu_count()}",
    "--parallel-compile",
    "--technique=convtuner",
    "--conv2d", "(576, 1280, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros')",
    "--input", "(1, 576, 1, 1)",
])))

opentuner.search.technique.register(AUCBanditMetaTechnique([
    # TODO(jansel): test more advanced search techniques
    NormalGreedyMutation(name="Normal5", mutation_rate=0.05),
    NormalGreedyMutation(name="Normal10", mutation_rate=0.10),
    NormalGreedyMutation(name="Normal15", mutation_rate=0.15),
    UniformGreedyMutation(name="Uniform50", mutation_rate=0.50),
], name="convtuner"))


class ConvTuner(MeasurementInterface):
    def __init__(self, args=None):
        manipulator = ConfigurationManipulator()
        conv_args = ast.literal_eval(args.conv2d)
        input_shape = ast.literal_eval(args.input)
        self.conv = torch.nn.Conv2d(*conv_args)
        self.image = torch.randn(*input_shape)

        # Dry run codegen to capture config
        cg = ConvolutionGenerator(ConfigRecorder(manipulator), self.conv, self.image, subproc=False)
        assert len(manipulator.params)
        # import pprint; pprint.pprint(manipulator.random())
        sec = float(median(timeit.repeat(lambda: cg(self.image), number=1, repeat=10)))
        self.times = max(1, int(0.01 / sec))

        super(ConvTuner, self).__init__(
            args=args,
            project_name="ConvTuner",
            program_name=repr(conv_args),
            program_version=repr(input_shape),
            manipulator=manipulator,
        )

    def seed_configurations(self):
        def min_value(p):
            if isinstance(p, NumericParameter):
                return p.min_value
            if isinstance(p, BooleanParameter):
                return False
            if isinstance(p, PermutationParameter):
                return list(p._items)
            assert False

        return [
            # {p.name: min_value(p) for p in node.manipulator().params}
        ]

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
        os.path.exists("../configs") or os.mkdir("../configs")
        return f"./configs/{self.program_name()},{self.program_version()}.json"


def main(args):
    multiprocessing.set_start_method("spawn")
    opentuner.init_logging()
    ConvTuner.main(parser.parse_args(args))
