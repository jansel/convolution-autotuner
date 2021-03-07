import argparse
import csv
import json
import logging
import os
import sys
import timeit
from ast import literal_eval
from collections import Counter
from subprocess import check_call

import pandas as pd
import torch
import torch.nn
from numpy import median

import convtuner.utils
from convtuner import codegen
from convtuner.config_recorder import ConfigProxy, DummyConfig
from convtuner.utils import gflops, Once, retry

CASES = [
    ((576, 1280, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros'),
     (1, 576, 1, 1)),  # 0
    ((16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros'),
     (32, 16, 55, 55)),  # 1
    ((512, 512, (3, 3), (2, 2), (1, 1), (1, 1), 32, False, 'zeros'),
     (32, 512, 28, 28)),  # 2
    ((128, 128, (3, 3), (1, 1), (1, 1), (1, 1), 32, False, 'zeros'),
     (32, 128, 56, 56)),  # 3
    ((120, 120, (5, 5), (1, 1), (2, 2), (1, 1), 120, False, 'zeros'),
     (32, 120, 28, 28)),  # 4
    ((64, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, False, 'zeros'),
     (32, 64, 56, 56)),  # 5
    ((512, 512, (3, 3), (2, 2), (1, 1), (1, 1), 32, False, 'zeros'),
     (32, 512, 28, 28)),  # 6
    ((120, 120, (5, 5), (1, 1), (2, 2), (1, 1), 120, False, 'zeros'),
     (32, 120, 28, 28)),  # 7
    ((128, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, False, 'zeros'),
     (32, 128, 28, 28)),  # 8
    ((16, 16, (3, 3), (2, 2), (1, 1), (1, 1), 16, False, 'zeros'),
    (1, 16, 112, 112)),  # 9
]
log = logging.getLogger(__name__)
stats = Counter()
results = []


def measure_testcase(cfg, conv: torch.nn.Conv2d, image: torch.Tensor):
    cg = codegen.ConvolutionGenerator(cfg, conv, image)
    result = cg(image)
    correct = conv(image)
    torch.testing.assert_allclose(correct, result)

    sec1 = median(timeit.repeat(lambda: conv(image), number=args.times, repeat=args.repeat))
    sec2 = median(timeit.repeat(lambda: cg(image), number=args.times, repeat=args.repeat))
    gf = gflops(conv, image) * args.times
    cg.close()
    return gf / sec1, gf / sec2, sec1 / sec2


def report_testcase(conv_args, input_shape):
    print(f"{conv_args}/{input_shape}")
    if args.dummy:
        cfg = DummyConfig()
    else:
        filename = f"./configs/{repr(conv_args)},{repr(input_shape)}.json"
        if args.autotune or not os.path.exists(filename):
            check_call([
                sys.executable,
                sys.argv[0],
                "autotune",
                "--conv2d", repr(conv_args),
                "--input", repr(input_shape),
                f"--test-limit={args.test_limit}",
            ])
        cfg = ConfigProxy(json.load(open(filename)))
    pytorch, autotuned, speedup = measure_testcase(
        cfg, torch.nn.Conv2d(*conv_args), torch.randn(input_shape))
    print(f"{pytorch:.1f} gflops => {autotuned:.1f} gflops ({speedup:.2f}x)")
    results.append([repr(conv_args), repr(input_shape),
                    f"{pytorch:.4f}", f"{autotuned:.4f}", f"{speedup:.4f}"])
    if speedup >= 1:
        stats["speedup_count"] += 1
        stats["speedup_factor"] += speedup
    stats["total"] += 1
    sys.stdout.flush()


def main(argv=None):
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--times', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--case', type=int)
    parser.add_argument('--autotune', action="store_true")
    parser.add_argument('--dummy', action="store_true")
    parser.add_argument('--limit', '-l', default=50, type=int)
    parser.add_argument('--test-limit', default=500, type=int)
    parser.add_argument('--testcases', default="testcases.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    codegen.VERBOSE = args.verbose

    if args.case is not None:
        report_testcase(*CASES[args.case])
        return

    with convtuner.utils.timer("total"):
        first = Once()
        testcases = pd.read_csv(args.testcases).sort_values("gflops")
        for _, row in testcases.iterrows():
            conv_args = literal_eval(row["conv2d"])
            input_shape = literal_eval(row["input"])
            if first(conv_args, input_shape):
                report_testcase(conv_args, input_shape)
            if len(first) >= args.limit:
                break

    stats["speedup_factor"] /= max(1, stats["speedup_count"])

    with open("results.csv", "w") as fd:
        csv.writer(fd).writerows(
            [["conv2d", "input", "pytorch_gflops", "autotuner_gflops", "speedup"]] +
            results)

    log.info("STATS %s", [f"{k}:{v}" for k, v in sorted(stats.items())])
    log.info("TIMERS %s", [f"{k}:{v:.2f}" for k, v in convtuner.utils.timers.most_common()])
