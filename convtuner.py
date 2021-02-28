#!/usr/bin/env python
import argparse
import contextlib
import csv
import ctypes
import gc
import logging
import os
import re
import sys
import time
import timeit
from ast import literal_eval
from collections import Counter
from _ctypes import dlclose
from functools import partial, reduce, cmp_to_key, lru_cache
from itertools import chain
from operator import mul
from subprocess import Popen, check_call, PIPE
from tempfile import NamedTemporaryFile
from typing import List

import pandas as pd
import sympy
import torch
import torch.utils.cpp_extension
from numpy import median
from sympy import Equality

S = partial(sympy.simplify, evaluate=False)
E = partial(sympy.simplify, evaluate=True)
VERBOSE = True
VECTOR_SIZE = 8
VECTOR_BITS = VECTOR_SIZE * 32
PREFIX_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix.h")
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
]

torch.set_num_threads(1)
log = logging.getLogger(__name__)
product = partial(reduce, mul)
stats = Counter()
timers = Counter()
results = []
args = None


@cmp_to_key
def cmp_expr(a, b):
    if is_true(S(a < b)):
        return -1
    elif is_true(S(a > b)):
        return 1
    else:
        assert str(a) == str(b)
        return 0


@lru_cache(128)
def is_true(x):
    return isinstance(E(nonnegative(x)), sympy.logic.boolalg.BooleanTrue)


@lru_cache(128)
def is_false(x):
    return isinstance(E(nonnegative(x)), sympy.logic.boolalg.BooleanFalse)


@lru_cache(128)
def is_boolean(x):
    return isinstance(E(nonnegative(x)),
                      (sympy.logic.boolalg.BooleanFalse, sympy.logic.boolalg.BooleanTrue))


def identity(x):
    return x


def nonnegative(x):
    return x.subs({v: sympy.Symbol(v, nonnegative=True, integer=True) for v in map(str, x.free_symbols)})


def tmpvar():
    n = f"tmp{tmpvar.count}"
    tmpvar.count += 1
    return n


@contextlib.contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timers[name] += t1 - t0


class Node(object):
    def visit(self, node_fn, expr_fn=identity):
        def wrap1(n):
            node_fn(n)
            return n

        def wrap2(n):
            expr_fn(n)
            return n

        self.apply(wrap1, wrap2)

    def apply_node(self, node_fn):
        return self.apply(node_fn, identity)

    def apply_expr(self, expr_fn):
        return self.apply(identity, expr_fn)

    def subs(self, replacements):
        return self.apply_expr(lambda x: x.subs(replacements))

    def replace_nodes(self, replacements):
        def visit(n):
            try:
                return replacements[n]
            except KeyError:
                return n

        return self.apply_node(visit)

    def apply(self, node_fn, expr_fn):
        raise NotImplementedError()

    def find_all(self, cls):
        result = []

        def visitor(n):
            if isinstance(n, cls):
                result.append(n)

        self.visit(visitor)
        return result

    def simplify(self):
        return self.apply(identity, identity)

    copy = simplify

    def simplify_conditionals(self, first, last):
        return self


class LoopRange(Node):
    def __str__(self):
        return f"for(int {self.name} = {self.begin}; {self.name} < {self.end}; {self.name} += {self.step})"

    def __init__(self, name, begin, end=None, step=1):
        super(LoopRange, self).__init__()
        self.name = name
        if end is None:  # mimic range()
            begin, end = 0, begin
        self.begin = S(begin)
        self.end = S(end)
        self.step = S(step)

    def apply(self, node_fn, expr_fn):
        return node_fn(LoopRange(name=self.name,
                                 begin=expr_fn(self.begin),
                                 end=expr_fn(self.end),
                                 step=expr_fn(self.step)))

    def can_vectorize(self):
        return (is_true(self.begin + VECTOR_SIZE <= self.end) and
                S(((self.end - self.begin) // self.step) % VECTOR_SIZE) == 0)


class Loops(Node):
    def __str__(self):
        return "\n".join(map(str, self.ranges + [self.body]))

    def __init__(self, ranges, body):
        super(Loops, self).__init__()
        self.ranges: List[LoopRange] = list(ranges)
        self.body: Node = body

    def apply(self, node_fn, expr_fn):
        body = self.body.apply(node_fn, expr_fn)
        if body is None:
            return None
        new_ranges = []
        for r in self.ranges:
            r = r.apply(node_fn, expr_fn)
            if (is_false(S(r.begin + r.step < r.end)) and
                    is_true(S(r.begin < r.end))):
                body = body.subs({r.name: r.begin})
            else:
                new_ranges.append(r)
        return node_fn(Loops(new_ranges, body))

    def simplify_conditionals(self, first, last):
        first = dict(first)
        last = dict(last)
        for rng in self.ranges:
            first[rng.name] = S(rng.begin.subs(first))
            last[rng.name] = S((rng.end - 1).subs(last))
        return Loops(self.ranges, self.body.simplify_conditionals(first, last))

    def cache_writes(self):
        body, stores = self.body.cache_writes()
        stores = set(stores)
        ranges = []
        seen = []
        defs = []
        sums = []

        def do_caching(store):
            nonlocal defs, sums, body, seen

            def matches(n):
                return store.name == n.name and is_true(Equality(store.index, n.index))

            if any(map(matches, seen)):
                return
            seen.append(store)
            var = tmpvar()
            if isinstance(store, ReduceStore):
                defs.append(TempDef(var).vectorize())
            else:
                defs.append(TempDef(var))
            sums.append(Sum(store, [TempRef(var)], "v0"))

            def swap(n):
                if isinstance(n, (Store, ReduceStore)) and matches(n):
                    return TempRef(var)
                return n

            body = body.apply_node(swap)

        for rng in reversed(self.ranges):
            stores_idep = {s for s in stores if sympy.diff(s.index, rng.name) == 0}
            if ranges:
                for store in stores - stores_idep:
                    do_caching(store)
            stores = stores_idep

            if defs or sums:
                body = Block(defs + [Loops(ranges, body)] + sums)
                ranges = [rng]
                defs.clear()
                sums.clear()
            else:
                ranges.insert(0, rng)

        for store in stores:
            do_caching(store)
        if defs or sums:
            return Block(defs + [Loops(ranges, body)] + sums), []
        else:
            return Loops(ranges, body), []

    def vectorize_loops(self):
        body, memory, banned = self.body.vectorize_loops()
        if any(m is False for m in memory):
            return self, [False], banned
        assert all(isinstance(x, str) for x in banned)
        assert all(isinstance(x, Memory) for x in memory)
        ranges = []
        first = True
        for rng in reversed(self.ranges):
            diffs = [sympy.diff(s.index, rng.name) for s in memory]
            if (first and
                    str(rng.name) not in banned and
                    all((x == 0 or x == 1) for x in diffs) and
                    any(x == 1 for x in diffs) and
                    rng.can_vectorize()):
                def swap(n):
                    if isinstance(n, Memory):
                        delta = sympy.diff(n.index, rng.name)
                        if delta == 1:
                            return n.vectorize()
                        assert delta == 0
                        return n.broadcast()
                    return n

                body = body.apply_node(swap)
                ranges.append(LoopRange(
                    rng.name,
                    rng.begin,
                    rng.end,
                    rng.step * VECTOR_SIZE
                ))
                first = False
            else:
                ranges.append(rng)
        if not first:
            memory = [False]  # stop
        return Loops(reversed(ranges), body), memory, banned

    def tiling_and_reorder(self):
        ranges = {}
        options0 = []
        options1 = []
        body = self.body
        for r in self.ranges:
            width = S(r.end - r.begin)
            assert r.begin == 0  # TODO(jansel): support this
            assert r.step == 1  # TODO(jansel): support this
            assert width.is_integer  # TODO(jansel): support this
            width = int(width)
            var = str(r.name)
            tiling = 16
            options0.append(var)

            if width % tiling == 0 and tiling > 1:
                options1.append(var)
                t0 = f"{var}_t0"
                t1 = f"{var}_t1"
                ranges[var] = [LoopRange(t0, width // tiling),
                               LoopRange(t1, tiling)]
                body = body.subs({var: S(f"{t0} * {tiling} + {t1}")})
            else:
                ranges[var] = [r]

        loop_order = list(chain(reversed(options0),
                                reversed(options1)))

        new_ranges = []
        for var in loop_order:
            new_ranges.append(ranges[var].pop())
        return Loops(reversed(new_ranges), body)

    def split_loops(self, constants, limit=9, threshold=8):
        conds = list(chain(*[c.tests for c in self.find_all(Condition)]))
        assert not any(map(is_boolean, conds))
        ranges = []
        body = self.body
        search = (constants["padding0"], 1)
        for rng in reversed(self.ranges):
            splits = []
            if limit >= 1 and not is_false((rng.end - rng.begin) // rng.step >= threshold):
                for offset in search:
                    idx = rng.begin + offset * rng.step
                    if (limit >= 1 and is_true(idx < rng.end) and
                            any(is_boolean(x.subs({rng.name: idx})) for x in conds)):
                        splits.append(idx)
                        break

                for offset in search:
                    idx = rng.begin + ((rng.end - rng.begin) // rng.step - offset) * rng.step
                    if (limit >= 1 and is_true(idx >= rng.begin) and
                            any(is_boolean(x.subs({rng.name: idx})) for x in conds)):
                        splits.append(idx)
                        break
            if splits:
                assert len(splits) == 1 or is_true(splits[0] < splits[1])
                split_ranges = []
                for split in splits:
                    split_ranges.append(LoopRange(
                        rng.name, rng.begin, split, rng.step
                    ))
                    rng = LoopRange(
                        rng.name, split, rng.end, rng.step
                    )
                split_ranges.append(rng)
                body = Loops(ranges, body)
                body = Block([Loops([x], body.copy()) for x in split_ranges])
                ranges.clear()
                limit /= len(splits)
                continue
            ranges.insert(0, rng)
        return Loops(ranges, body)


class Condition(Node):
    def __str__(self):
        test = " && ".join(map(str, self.tests))
        return f"if({test})\n{self.body}"

    def __init__(self, tests, body):
        super(Condition, self).__init__()
        self.tests: List[sympy.Expr] = list(map(S, tests))
        self.body: Node = body

    def apply(self, node_fn, expr_fn):
        tests = [expr_fn(x) for x in self.tests]
        tests = [x for x in tests if not is_true(x)]
        body = self.body.apply(node_fn, expr_fn)
        if not tests:
            return body
        if any(map(is_false, tests)):
            return None
        return node_fn(Condition(tests, body))

    def simplify_conditionals(self, first_sub, last_sub):
        tests = []
        for t in self.tests:
            # this assumes conditionals are monotonic
            first = t.subs(first_sub)
            last = t.subs(last_sub)
            if is_true(first) and is_true(last):
                tests.append(S(True))
            elif is_false(first) and is_false(last):
                tests.append(S(False))
            else:
                tests.append(t)
        return Condition(tests, self.body.simplify_conditionals(first, last)).simplify()

    def cache_writes(self):
        body, stores = self.body.cache_writes()
        return Condition(self.tests, body), stores

    def vectorize_loops(self):
        body, memory, banned = self.body.vectorize_loops()
        banned = reduce(set.union, [banned] + [
            set(map(str, t.free_symbols)) for t in self.tests
        ])
        return Condition(self.tests, body), memory, banned


class Reduction(Node):
    def __init__(self, output, inputs, expression):
        super(Reduction, self).__init__()
        self.output: Memory = output
        self.inputs: List[Memory] = list(inputs)
        self.expression: sympy.Expr = S(expression)

    def apply(self, node_fn, expr_fn):
        output = self.output.apply(node_fn, expr_fn)
        inputs = [x.apply(node_fn, expr_fn) for x in self.inputs]
        expression = expr_fn(self.expression)
        return node_fn(self.__class__(output, inputs, expression))

    def cache_writes(self):
        return self, self.find_all((Store, ReduceStore))

    def vectorize_loops(self):
        return self, self.find_all(Memory), set()


class Sum(Reduction):
    def __str__(self):
        expr = str(self.expression)
        for i, v in enumerate(self.inputs):
            expr = re.sub(fr"\bv{i}\b", str(v), expr)
        if isinstance(self.output, ReduceStore):
            return f"{self.output} += _mm{VECTOR_BITS}_reduce_add_ps({expr});"
        return f"{self.output} += {expr};"


class Block(Node):
    def __init__(self, statements):
        super(Block, self).__init__()
        self.statements = statements

    def apply(self, node_fn, expr_fn):
        stmts = []
        for s in self.statements:
            s = s.apply(node_fn, expr_fn)
            if isinstance(s, Block):
                stmts.extend(s.statements)
            elif s is not None:
                stmts.append(s)
        if len(stmts) == 0:
            return None
        if len(stmts) == 1:
            return stmts[0]
        return node_fn(Block(stmts))

    def __str__(self):
        return "{\n" + "\n".join(map(str, self.statements)) + "}\n"

    def simplify_conditionals(self, first, last):
        return Block([x.simplify_conditionals(first, last)
                      for x in self.statements])

    def cache_writes(self):
        statements = []
        stores = []
        for s in self.statements:
            a, b = s.cache_writes()
            statements.append(a)
            stores.extend(b)
        return Block(statements), stores

    def vectorize_loops(self):
        statements = []
        memory = []
        banned = set()
        for s in self.statements:
            a, b, c = s.vectorize_loops()
            statements.append(a)
            memory.extend(b)
            banned.update(c)
        return Block(statements), memory, banned


class Memory(Node):
    @classmethod
    def from_indices(cls, name, indices):
        assert isinstance(indices, list)
        return cls(name, sum(sympy.Mul(S(v), S(f"{name}_stride{i}"))
                             for i, v in enumerate(indices)))

    def __init__(self, name, index):
        super(Memory, self).__init__()
        self.name = name
        self.index = S(index)

    def __str__(self):
        return f"{self.name}[{self.index}]"

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(self.name, expr_fn(self.index)))


class Load(Memory):
    def broadcast(self):
        return BroadcastLoad(self.name, self.index)

    def vectorize(self):
        return VectorLoad(self.name, self.index)


class Store(Memory):
    def broadcast(self):
        return ReduceStore(self.name, self.index)

    def vectorize(self):
        return VectorStore(self.name, self.index)


class VectorizedMemory(Memory):
    def __str__(self):
        return f"(*(__m{VECTOR_BITS}* __restrict__)({self.name} + {self.index}))"


class VectorLoad(VectorizedMemory):
    pass


class VectorStore(VectorizedMemory):
    pass


class ReduceStore(Memory):
    pass


class BroadcastLoad(Memory):
    def __str__(self):
        return f"_mm{VECTOR_BITS}_broadcast_ss({self.name} + {self.index})"


class TempRef(Node):
    def __init__(self, name):
        super(TempRef, self).__init__()
        self.name = name

    def __str__(self):
        return self.name

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(self.name))


class TempDef(Node):
    def __init__(self, name, dtype="float", init="0"):
        super(TempDef, self).__init__()
        self.dtype = dtype
        self.name = name
        self.init = init

    def __str__(self):
        return f"{self.dtype} {self.name} = {self.init};"

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(self.name, self.dtype, self.init))

    def vectorize(self):
        assert self.dtype == "float" and self.init == "0"
        return self.__class__(self.name,
                              f"__m{VECTOR_BITS}",
                              f"_mm{VECTOR_BITS}_setzero_ps()")


class ConvolutionGenerator(object):
    @staticmethod
    def get_constants(conv, image):
        # These depend on conv
        out_channels = conv.out_channels
        padding = conv.padding
        dilation = conv.dilation
        kernel_size = conv.kernel_size
        stride = conv.stride
        in_channels = conv.in_channels
        groups = conv.groups
        weight = conv.weight
        stride0, stride1 = stride
        dilation0, dilation1 = dilation
        padding0, padding1 = padding
        kernel_size0, kernel_size1 = kernel_size
        assert conv.padding_mode == "zeros"

        # These depend on conv + image, plan to make these dynamic in the future
        batch_size, _, *in_sizes = image.shape
        out_sizes = [
            (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
            for i, v in enumerate(in_sizes)]
        in_sizes0, in_sizes1 = in_sizes
        out_sizes0, out_sizes1 = out_sizes

        out = torch.zeros([batch_size, out_channels,
                           out_sizes0, out_sizes1], dtype=image.dtype)

        # TODO(jansel): make these dynamic
        image_stride0, image_stride1, image_stride2, image_stride3 = image.stride()
        weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
        out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

        return {k: v for k, v in locals().items() if isinstance(v, int)}

    def __init__(self, conv, image):
        super(ConvolutionGenerator, self).__init__()
        self.constants = self.get_constants(conv, image)
        self.dtype = "float"
        self.weight = conv.weight
        self.bias = conv.bias
        self.out_shape = (self.constants["batch_size"],
                          self.constants["out_channels"],
                          self.constants["out_sizes0"],
                          self.constants["out_sizes1"])
        if self.bias is not None:
            self.bias = self.bias.view(1, self.bias.shape[0], 1, 1)
        tmpvar.count = 0

        macros = {"in_channel": S("in_channel_g + g * (in_channels // groups)"),
                  "out_channel": S("out_channel_g + g * (out_channels // groups)"),
                  "in0": S("out0 * stride0 + kernel0 * dilation0 - padding0"),
                  "in1": S("out1 * stride1 + kernel1 * dilation1 - padding1"), }

        # This defines the convolution algorithm
        self.body = Loops(
            [LoopRange("n", "batch_size"),
             LoopRange("g", "groups"),
             LoopRange("out_channel_g", "out_channels // groups"),
             LoopRange("in_channel_g", "in_channels // groups"),
             LoopRange("out0", "out_sizes0"),
             LoopRange("out1", "out_sizes1"),
             LoopRange("kernel0", "kernel_size0"),
             LoopRange("kernel1", "kernel_size1")],
            Condition(
                ["0 <= in0",
                 "0 <= in1",
                 "in0 < in_sizes0",
                 "in1 < in_sizes1"],
                Sum(
                    Store.from_indices("out", ["n", "out_channel", "out0", "out1"]),
                    [Load.from_indices("weight", ["out_channel", "in_channel_g", "kernel0", "kernel1"]),
                     Load.from_indices("image", ["n", "in_channel", "in0", "in1"])],
                    "v0 * v1"))).subs(macros).subs(self.constants)

        # Some ad-hoc compiler passes
        with timer("simplify"):
            self.body = self.body.simplify_conditionals(dict(), dict())
        with timer("tiling_and_reorder"):
            self.body = self.body.tiling_and_reorder()
        with timer("split_loops"):
            self.body = self.body.split_loops(self.constants)
        with timer("simplify"):
            self.body = self.body.simplify_conditionals(dict(), dict())
        with timer("vectorize"):
            self.body, _, _ = self.body.vectorize_loops()
        with timer("cache_writes"):
            self.body, _ = self.body.cache_writes()

        self.print()

        with NamedTemporaryFile(suffix=".so") as shared_object:
            with NamedTemporaryFile(suffix=".cpp") as source:
                source.write(str(self).encode("utf-8"))
                source.flush()
                cmd = [
                    args.cxx, "-shared", "-o", shared_object.name, source.name,
                    f"-O{args.opt}", "-ffast-math", "-march=native",
                    "-Wall", "-Werror"
                ]
                log.debug(" ".join(cmd))
                with timer("compile"):
                    check_call(cmd)
            lib = ctypes.cdll.LoadLibrary(shared_object.name)
            self.compiled = lib.convolution
            self.handle = lib._handle

    def close(self):
        if self.compiled is not None:
            self.compiled = None
            gc.collect()
            dlclose(self.handle)
            self.handle = None
            gc.collect()

    def print(self):
        VERBOSE > 1 and Popen("clang-format", stdin=PIPE).communicate(str(self).encode("utf-8"))

    def __str__(self):
        return f"""
            #include "{PREFIX_HEADER}"
            extern "C" void convolution(
                float* __restrict__ image,
                float* __restrict__ weight,
                float* __restrict__ out
            )
            {{
                {self.body}
            }}
            """

    def __call__(self, image):
        out = torch.zeros(self.out_shape)
        self.compiled(ctypes.c_void_p(image.data_ptr()),
                      ctypes.c_void_p(self.weight.data_ptr()),
                      ctypes.c_void_p(out.data_ptr()))
        if self.bias is not None:
            out += self.bias
        return out


def gflops(conv, image):
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    groups = conv.groups
    padding = conv.padding
    dilation = conv.dilation
    kernel_size = list(conv.kernel_size)
    stride = conv.stride
    bias = conv.bias
    batch_size, _, *in_sizes = image.shape
    out_sizes = [
        (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        for i, v in enumerate(in_sizes)]
    gflops = product([2,
                      batch_size,
                      groups,
                      out_channels // groups,
                      in_channels // groups] +
                     out_sizes +
                     kernel_size)
    if bias is not None:
        gflops += product([batch_size, out_channels] + out_sizes)
    return gflops / 1000000000.0


def measure_testcase(conv: torch.nn.Conv2d, image: torch.Tensor):
    cg = ConvolutionGenerator(conv, image)
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
    pytorch, autotuned, speedup = measure_testcase(torch.nn.Conv2d(*conv_args), torch.randn(input_shape))
    print(f"{pytorch:.1f} gflops => {autotuned:.1f} gflops ({speedup:.2f}x)")
    results.append([repr(conv_args), repr(input_shape),
                    f"{pytorch:.4f}", f"{autotuned:.4f}", f"{speedup:.4f}"])
    if speedup >= 1:
        stats["speedup_count"] += 1
        stats["speedup_factor"] += speedup
    stats["total"] += 1
    sys.stdout.flush()


class Once(set):
    def __call__(self, *x):
        return x not in self and (self.add(x) or True)


def unittests():
    measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    measure_testcase(torch.nn.Conv2d(8, 4, (3, 1), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1), torch.randn(1, 8, 8, 8))
    measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=8), torch.randn(2, 8, 8, 8))
    measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=2), torch.randn(2, 8, 8, 8))
    # TODO(jansel): debug these ones
    # measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1, dilation=2), torch.randn(2, 8, 8, 8))
    # measure_testcase(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=5, groups=4, bias=False), torch.randn(2, 8, 16, 8))


def main(argv=None):
    global args, VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--times', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--unittest', action="store_true")
    parser.add_argument('--case', type=int)
    parser.add_argument('--limit', '-l', default=9999, type=int)
    parser.add_argument('--testcases', default="testcases.csv")
    parser.add_argument('--cxx', default="gcc-10")
    parser.add_argument('--opt', '-O', type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    VERBOSE = args.verbose

    if args.unittest:
        unittests()
        return

    if args.case is not None:
        report_testcase(*CASES[args.case])
        return

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
    log.info("TIMERS %s", [f"{k}:{v:.2f}" for k, v in timers.most_common()])
    # check_call(["cat", f"/proc/{os.getpid()}/maps"])


if __name__ == "__main__":
    main()
