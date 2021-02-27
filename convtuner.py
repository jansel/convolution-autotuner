#!/usr/bin/env python
import re
import subprocess
import timeit
from ast import literal_eval
from functools import partial, reduce
from itertools import chain
from operator import mul
from typing import List

import pandas as pd
import sympy
import torch
import torch.utils.cpp_extension

torch.set_num_threads(1)

product = partial(reduce, mul)
S = partial(sympy.simplify, evaluate=True)
VERBOSE = True
VECTOR_SIZE = 8
VECTOR_BITS = VECTOR_SIZE * 32
TIMES = 3


def is_true(x):
    return isinstance(S(x), sympy.logic.boolalg.BooleanTrue)


def is_false(x):
    return isinstance(S(x), sympy.logic.boolalg.BooleanFalse)


def identity(x):
    return x


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


class LoopRange(Node):
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

    def __str__(self):
        return f"for(int {self.name} = {self.begin}; {self.name} < {self.end}; {self.name} += {self.step})"


class Loops(Node):
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

    def __str__(self):
        return "\n".join(map(str, self.ranges + [self.body]))


class Condition(Node):
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

    def __str__(self):
        test = " && ".join(map(str, self.tests))
        return f"if({test})\n{self.body}"


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
    counter = 0

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

        image_stride0, image_stride1, image_stride2, image_stride3 = image.stride()
        weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
        out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

        return {k: v for k, v in locals().items() if isinstance(v, int)}

    def __init__(self, conv, image):
        super(ConvolutionGenerator, self).__init__()
        self.name = f"conv{self.counter}"
        ConvolutionGenerator.counter += 1
        self.constants = self.get_constants(conv, image)
        self.dtype = "float"
        self.weight = conv.weight
        self.bias = conv.bias
        if self.bias is not None:
            self.bias = self.bias.view(1, self.bias.shape[0], 1, 1)
        self._tmpvar = 0

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

        self.run_pass(self.simplify_conditionals_pass)
        self.run_pass(self.vectorize_pass)
        self.run_pass(self.cache_writes_pass)
        self.print()

        torch.utils.cpp_extension.load_inline(
            name=self.name,
            cpp_sources=str(self),
            is_python_module=False,
            extra_cflags=['-O3', '-ffast-math', '-march=native'],
            verbose=False,
        )
        self.compiled = getattr(getattr(torch.ops, self.name), self.name)

    def print(self):
        VERBOSE and subprocess.Popen("clang-format", stdin=subprocess.PIPE).communicate(str(self).encode("utf-8"))

    def tmpvar(self):
        n = f"tmp{self._tmpvar}"
        self._tmpvar += 1
        return n

    def run_pass(self, pass_fn, target_cls=Loops):
        def wrapper(node):
            if isinstance(node, target_cls):
                return pass_fn(node)
            return node

        self.print()
        VERBOSE and print(pass_fn.__qualname__)
        self.body = self.body.apply_node(wrapper)

    def vectorize_pass(self, loops: Loops, direction=reversed):
        memory = loops.find_all(Memory)
        conds = reduce(set.union,
                       [set(map(str, t.free_symbols)) for t in chain(*[c.tests for c in loops.find_all(Condition)])],
                       set())
        ranges = []
        body = loops.body
        first = True
        for rng in direction(loops.ranges):
            diffs = [sympy.diff(s.index, rng.name) for s in memory]
            if (first and
                    rng.name not in conds and
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
        return Loops(direction(ranges), body)

    def cache_writes_pass(self, loops: Loops):
        stores = set(chain(*[s.find_all((Store, ReduceStore)) for s in loops.find_all(Sum)]))
        ranges = []
        body = loops.body
        for rng in reversed(loops.ranges):
            stores_idep = {s for s in stores if sympy.diff(s.index, rng.name) == 0}
            defs = []
            sums = []
            if ranges:
                for store in stores - stores_idep:
                    var = self.tmpvar()
                    if isinstance(store, ReduceStore):
                        defs.append(TempDef(var).vectorize())
                    else:
                        defs.append(TempDef(var))
                    sums.append(Sum(store, [TempRef(var)], "v0"))

                    def swap(n):
                        if isinstance(n, (Store, ReduceStore)) and str(store) == str(n):
                            return TempRef(var)
                        return n

                    body = body.apply_node(swap)
            if defs or sums:
                body = Block(defs + [Loops(ranges, body)] + sums)
                ranges = [rng]
            else:
                ranges.insert(0, rng)
            stores = stores_idep
        return Loops(ranges, body)

    def simplify_conditionals_pass(self, loops: Loops):
        first_value = {}
        last_value = {}
        for rng in loops.ranges:
            first_value[rng.name] = S(rng.begin.subs(first_value))
            last_value[rng.name] = S((rng.end - 1).subs(last_value))

        def visit(node):
            if isinstance(node, Condition):
                tests = []
                for t in node.tests:
                    # this pass assumes conditionals are monotonic
                    first = S(t.subs(first_value))
                    last = S(t.subs(last_value))
                    if is_true(first) and is_true(last):
                        tests.append(S(True))
                    elif is_false(first) and is_false(last):
                        tests.append(S(False))
                    else:
                        tests.append(t)
                return Condition(tests, node.body).apply_expr(identity)
            return node

        return loops.apply_node(visit)

    def __str__(self):
        return f"""
            #include <torch/script.h>
            #include <immintrin.h>
            static inline float _mm256_reduce_add_ps(__m256 x) {{
                const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
                const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
                const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
                return _mm_cvtss_f32(x32);
            }}
            torch::Tensor {self.name}(
                    const torch::Tensor& _image,
                    const torch::Tensor& _weight)
            {{
                torch::Tensor _out = torch::zeros({{
                    {self.constants["batch_size"]},
                    {self.constants["out_channels"]},
                    {self.constants["out_sizes0"]},
                    {self.constants["out_sizes1"]}
                }}); 
                float* __restrict__ image = _image.data_ptr<float>();
                float* __restrict__ weight = _weight.data_ptr<float>();
                float* __restrict__ out = _out.data_ptr<float>();
                {self.body}
                return _out;
            }}
            TORCH_LIBRARY({self.name}, m)
            {{
                m.def("{self.name}", &{self.name});
            }}
            """

    def __call__(self, image):
        out = self.compiled(image, self.weight)
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


def test_conv(conv: torch.nn.Conv2d, image: torch.Tensor):
    cg = ConvolutionGenerator(conv, image)
    result = cg(image)
    correct = conv(image)
    torch.testing.assert_allclose(correct, result)

    sec1 = timeit.timeit(lambda: conv(image), number=TIMES)
    sec2 = timeit.timeit(lambda: cg(image), number=TIMES)
    gf = gflops(conv, image) * TIMES
    print(f"{gf / sec1:.1f} gflops => {gf / sec2:.1f} gflops - {sec1 / sec2:.2f}x")


class Once(set):
    def __call__(self, *x):
        return x not in self and (self.add(x) or True)


def unittests():
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 4, (3, 1), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1), torch.randn(1, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1, dilation=2), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=8), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=2), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=5, groups=4, bias=False), torch.randn(2, 8, 16, 8))


def main():
    global VERBOSE
    VERBOSE = True

    if False:
        unittests()
        return

    if False:
        # ReduceStore
        test_conv(torch.nn.Conv2d(576, 1280, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros'),
                  torch.randn(1, 576, 1, 1))
        return

    if False:
        test_conv(torch.nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, 'zeros'),
                  torch.rand(32, 16, 55, 55))
        return

    if False:
        test_conv(torch.nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), (1, 1), 32, False, 'zeros'),
                  torch.randn(32, 512, 28, 28))

    if False:
        # BroadcastLoad / padding
        test_conv(
            torch.nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), (1, 1), 32, False, 'zeros'),
            torch.randn(32, 128, 56, 56)
        )
        return

    VERBOSE = False

    first = Once()
    testcases = pd.read_csv("testcases.csv").sort_values("gflops")
    for _, row in testcases.iterrows():
        conv_args = literal_eval(row["conv2d"])
        input_shape = literal_eval(row["input"])
        if first(conv_args, input_shape):
            print(f"{conv_args}/{input_shape}")
            test_conv(torch.nn.Conv2d(*conv_args),
                      torch.randn(input_shape))
        if len(first) > 50:
            return


if __name__ == "__main__":
    main()
