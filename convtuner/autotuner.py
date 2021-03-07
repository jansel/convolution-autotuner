import argparse
import ast
import json
import logging
import multiprocessing
import os
import random
import shutil
import timeit

import opentuner
import torch
from numpy import median
from opentuner import Result
from opentuner.measurement import MeasurementInterface
from opentuner.search.bandittechniques import AUCBanditMetaTechnique
from opentuner.search.evolutionarytechniques import NormalGreedyMutation
from opentuner.search.manipulator import BooleanParameter
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import EnumParameter
from opentuner.search.manipulator import NumericParameter
from opentuner.search.manipulator import PermutationParameter

from convtuner.codegen import ConvolutionGenerator
from convtuner.config_recorder import ConfigProxy
from convtuner.config_recorder import ConfigRecorder
from convtuner.utils import gflops, Once

log = logging.getLogger(__name__)
REPEAT = 5

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("--conv2d")
parser.add_argument("--input")
parser.set_defaults(**vars(parser.parse_args([
    "--no-dups",
    # "--stop-after=600",
    "--test-limit=1000",
    f"--parallelism={multiprocessing.cpu_count()}",
    "--parallel-compile",
    "--technique=convtuner",
    "--conv2d", "(576, 1280, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros')",
    "--input", "(1, 576, 1, 1)",
])))

# opentuner.search.technique.register(AUCBanditMetaTechnique([
#     NormalGreedyMutation(name="Normal5", mutation_rate=0.05),
#     NormalGreedyMutation(name="Normal10", mutation_rate=0.10),
#     DifferentialEvolution(population_size=10, cr=0.8, information_sharing=1),
#     PatternSearch(),
# ], name="convtuner"))
# DifferentialEvolution(population_size=10, cr=0.8, information_sharing=1),

# Used for finetuning
opentuner.search.technique.register(AUCBanditMetaTechnique([
    NormalGreedyMutation(name="Normal4", mutation_rate=0.04),
    NormalGreedyMutation(name="Normal8", mutation_rate=0.08),
    NormalGreedyMutation(name="Normal16", mutation_rate=0.16),
], name="convtuner"))


class ConvTuner(MeasurementInterface):
    def __init__(self, args=None):
        manipulator = ConfigurationManipulator()
        conv_args = ast.literal_eval(args.conv2d)
        input_shape = ast.literal_eval(args.input)
        self.conv = torch.nn.Conv2d(*conv_args)
        self.image = torch.randn(*input_shape)

        # Dry run codegen to capture config
        cg = ConvolutionGenerator(ConfigRecorder(manipulator), self.conv, self.image)
        assert len(manipulator.params)
        # import pprint; pprint.pprint(manipulator.random())
        sec = float(median(timeit.repeat(lambda: cg(self.image), number=1, repeat=REPEAT)))
        self.times = max(1, int(0.1 / sec))

        super(ConvTuner, self).__init__(
            args=args,
            project_name="ConvTuner",
            program_name=repr(conv_args),
            program_version=repr(input_shape),
            manipulator=manipulator,
        )

    def default_config(self):
        return {p.name: min_value(p) for p in self.manipulator().params}

    def seed_configurations(self):
        first = Once()
        seeds = [self.default_config()]
        for name in os.listdir("configs"):
            config = self.default_config()
            data = open(os.path.join("configs", name)).read()
            seed = json.loads(data)
            seed = {k: v for k, v in seed.items() if k in config}
            if not first(data):
                pass
            elif set(seed.keys()) == set(config.keys()):
                seeds.append(seed)
            else:  # New config values we need to initialize
                config2 = self.manipulator().random()
                config.update(seed)
                config2.update(seed)
                seeds.extend([config, config2])
        random.shuffle(seeds)
        return seeds[:50]

    def compile_and_run(self, desired_result, input, limit):
        return Result(time=self.measure_cfg(desired_result.configuration.data))

    def measure_cfg(self, cfg):
        cg = ConvolutionGenerator(ConfigProxy(cfg), self.conv, self.image)
        cg(self.image)  # warmup
        sec = median(timeit.repeat(lambda: cg(self.image), number=self.times, repeat=REPEAT))
        cg.close()
        return float(sec)

    def compile(self, cfg, id):
        filename = os.path.join("bins", str(id))
        assert not os.path.exists(filename)
        ConvolutionGenerator(ConfigProxy(cfg), self.conv, self.image,
                             standalone_filename=filename,
                             standalone_times=self.times)
        assert os.path.exists(filename)
        # filename now contains a test binary
        return "OK", filename

    def run_precompiled(self, desired_result, input, limit, compile_result,
                        result_id):
        compile_result, filename = compile_result
        assert compile_result == "OK"
        run_result = self.call_program([filename], limit=limit)
        os.unlink(filename)
        if run_result['returncode'] != 0:
            if run_result['timeout']:
                return Result(state='TIMEOUT', time=float('inf'))
            else:
                log.error('program error')
                return Result(state='ERROR', time=float('inf'))
        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        cfg = configuration.data
        sec = self.measure_cfg(cfg)
        gf = gflops(self.conv, self.image) * self.times / sec
        print(f"Final configuration ({sec:.2f}s, {gf:.1f} gflops): {self.config_filename()}")
        with open(self.config_filename(), "w") as fd:
            json.dump(cfg, fd, indent=2, sort_keys=True)
            fd.write("\n")

    def config_filename(self):
        os.path.exists("../configs") or os.mkdir("../configs")
        return f"./configs/{self.program_name()},{self.program_version()}.json"


def min_value(p):
    if isinstance(p, NumericParameter):
        return p.min_value
    if isinstance(p, BooleanParameter):
        return False
    if isinstance(p, PermutationParameter):
        return list(p._items)
    if isinstance(p, EnumParameter):
        return p.options[0]
    assert False


def main(args):
    os.path.exists("bins") or os.mkdir("bins")
    try:
        multiprocessing.set_start_method("spawn")
        opentuner.init_logging()
        ConvTuner.main(parser.parse_args(args))
    finally:
        shutil.rmtree("./bins")
