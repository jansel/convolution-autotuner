import subprocess
import timeit
from ast import literal_eval
from functools import partial, reduce
from operator import mul

import pandas as pd
import sympy
import torch
import torch.utils.cpp_extension
from typing import List
import re

torch.set_num_threads(1)

product = partial(reduce, mul)
S = partial(sympy.simplify, evaluate=True)
PRINT = False


def naive_conv(conv):
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    groups = conv.groups
    padding = conv.padding
    dilation = conv.dilation
    kernel_size = conv.kernel_size
    stride = conv.stride
    weight = conv.weight
    bias = conv.bias
    stride0, stride1 = stride
    dilation0, dilation1 = dilation
    padding0, padding1 = padding
    kernel_size0, kernel_size1 = kernel_size
    if bias is not None:
        bias = bias.view(1, bias.shape[0], 1, 1)
    assert conv.padding_mode == "zeros"

    def _conv(image):
        batch_size, _, *in_sizes = image.shape
        out_sizes = [
            (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
            for i, v in enumerate(in_sizes)]
        out = torch.zeros([batch_size, out_channels] + out_sizes)
        in_sizes0, in_sizes1 = in_sizes
        out_sizes0, out_sizes1 = out_sizes

        for n in range(batch_size):
            for group in range(groups):
                for out_channel_g in range(out_channels // groups):
                    for in_channel_g in range(in_channels // groups):
                        for out0 in range(out_sizes0):
                            for out1 in range(out_sizes1):
                                for kernel0 in range(kernel_size0):
                                    for kernel1 in range(kernel_size1):
                                        in_chan = in_channel_g + group * (in_channels // groups)
                                        out_chan = out_channel_g + group * (out_channels // groups)
                                        in0 = out0 * stride0 + kernel0 * dilation0 - padding0
                                        in1 = out1 * stride1 + kernel1 * dilation1 - padding1
                                        if 0 <= in0 < in_sizes0 and 0 <= in1 < in_sizes1:
                                            out[n, out_chan, out0, out1] += (
                                                    weight[out_chan, in_channel_g, kernel0, kernel1] *
                                                    image[n, in_chan, in0, in1]
                                            )

        if bias is not None:
            out += bias
        return out

    return _conv


class Node(object):
    pass


class LoopRange(Node):
    def __init__(self, name, begin, end=None, step=1):
        super(LoopRange, self).__init__()
        self.name = name
        if end is None:  # mimic range()
            begin, end = 0, begin
        self.begin = S(begin)
        self.end = S(end)
        self.step = S(step)

    def subs(self, replacements):
        self.begin = self.begin.subs(replacements)
        self.end = self.end.subs(replacements)
        self.step = self.step.subs(replacements)
        return self

    def __str__(self):
        return f"for(int {self.name} = {self.begin}; {self.name} < {self.end}; {self.name} += {self.step})"


class Loops(Node):
    def __init__(self, ranges, body):
        super(Loops, self).__init__()
        self.ranges: List[LoopRange] = list(ranges)
        self.body: Node = body

    def subs(self, replacements):
        self.body = self.body.subs(replacements)
        new_ranges = []
        for r in self.ranges:
            r = r.subs(replacements)
            if (isinstance(S(r.begin + r.step < r.end), sympy.logic.boolalg.BooleanFalse) and
                    isinstance(S(r.begin < r.end), sympy.logic.boolalg.BooleanTrue)):
                self.body = self.body.subs({r.name: r.begin})
            else:
                new_ranges.append(r)
        self.ranges = new_ranges
        return self

    def __str__(self):
        return "\n".join(map(str, self.ranges + [self.body]))


class Condition(Node):
    def __init__(self, tests, body):
        super(Condition, self).__init__()
        self.tests: List[sympy.Expr] = list(map(S, tests))
        self.body: Node = body

    def subs(self, replacements):
        self.tests = [x.subs(replacements) for x in self.tests]
        self.tests = [x for x in self.tests if not isinstance(x, sympy.logic.boolalg.BooleanTrue)]
        self.body = self.body.subs(replacements)
        return self

    def __str__(self):
        test = " && ".join(map(str, self.tests))
        return f"if({test})\n{self.body}"


class Sum(Node):  # could generalize to Reduction()
    def __init__(self, output, inputs, expression):
        super(Sum, self).__init__()
        self.output: Memory = output
        self.inputs: List[Memory] = list(inputs)
        self.expression: sympy.Expr = S(expression)

    def subs(self, replacements):
        self.output = self.output.subs(replacements)
        self.inputs = [x.subs(replacements) for x in self.inputs]
        self.expression = self.expression.subs(replacements)
        return self

    def __str__(self):
        expr = str(self.expression)
        for i, v in enumerate(self.inputs):
            expr = re.sub(fr"\bv{i}\b", str(v), expr)
        return f"{self.output} += {expr};"


class Memory(Node):
    def __init__(self, name, indices):
        super(Memory, self).__init__()
        self.name = name
        assert isinstance(indices, list)
        self.index = S(sum(sympy.Mul(S(v), S(f"{name}_stride{i}"))
                           for i, v in enumerate(indices)))

    def __str__(self):
        return f"{self.name}[{self.index}]"

    def subs(self, replacements):
        self.index = S(self.index.subs(replacements))
        return self


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
        self.dtype = image.dtype
        self.weight = conv.weight
        self.bias = conv.bias
        if self.bias is not None:
            self.bias = self.bias.view(1, self.bias.shape[0], 1, 1)

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
                    Memory("out", ["n", "out_channel", "out0", "out1"]),
                    [Memory("weight", ["out_channel", "in_channel_g", "kernel0", "kernel1"]),
                     Memory("image", ["n", "in_channel", "in0", "in1"])],
                    "v0 * v1"))).subs(macros).subs(self.constants)

        PRINT and subprocess.Popen("clang-format", stdin=subprocess.PIPE).communicate(str(self).encode("utf-8"))
        torch.utils.cpp_extension.load_inline(
            name=self.name,
            cpp_sources=str(self),
            is_python_module=False,
            extra_cflags=['-O3', '-ffast-math', '-march=native'],
            verbose=False,
        )
        self.compiled = getattr(getattr(torch.ops, self.name), self.name)

    def __str__(self):
        return f"""
            # include <torch/script.h>
            void {self.name}(const torch::Tensor& _weight,
                        const torch::Tensor& _image,
                        const torch::Tensor& _out) {{
                float* __restrict__ weight = _weight.data_ptr<float>();
                float* __restrict__ image = _image.data_ptr<float>();
                float* __restrict__ out = _out.data_ptr<float>();
                {self.body}
            }}
            TORCH_LIBRARY({self.name}, m)
            {{
                m.def("{self.name}", &{self.name});
            }}
            """

    def __call__(self, image):
        out = torch.zeros([self.constants["batch_size"],
                           self.constants["out_channels"],
                           self.constants["out_sizes0"],
                           self.constants["out_sizes1"]], dtype=image.dtype)
        self.compiled(self.weight, image, out)
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
    correct = conv(image)
    # result = naive_conv1(conv)(image)
    cg = ConvolutionGenerator(conv, image)
    result = cg(image)
    torch.testing.assert_allclose(correct, result)

    sec1 = timeit.timeit(lambda: conv(image), number=3)
    sec2 = timeit.timeit(lambda: cg(image), number=3)
    print(f"baseline {sec1 / gflops(conv, image):.1f}")
    print(f"generated {sec2 / gflops(conv, image):.1f}")
    print(f"speedup {sec1 / sec2:.2f}")


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
    # unittests()
    # return

    first = Once()
    testcases = pd.read_csv("testcases.csv").sort_values("gflops")
    for _, row in testcases.iterrows():
        conv_args = literal_eval(row["conv2d"])
        input_shape = literal_eval(row["input"])
        if first(conv_args, input_shape):
            print(conv_args, input_shape, )
            test_conv(torch.nn.Conv2d(*conv_args),
                      torch.randn(input_shape))
        if len(first) > 10:
            return


if __name__ == "__main__":
    main()
