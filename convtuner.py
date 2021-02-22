from ast import literal_eval
from functools import partial

import torch
import torch.utils.cpp_extension
import os
from timeit import default_timer as timer
import sympy
import pandas as pd
import itertools

simplify = partial(sympy.simplify, evaluate=True)

fwd_op_cpp = """
#include <torch/script.h>

torch::Tensor my_fwd(const torch::Tensor& ad_emb_packed, const torch::Tensor& user_emb, const torch::Tensor& wide, const torch::Tensor& mu, const torch::Tensor& sigma, const torch::Tensor& fc_b, const torch::Tensor& fc_w) {
  const auto& wide_offset = at::add(wide, mu);
  const auto& wide_normalized = at::mul(wide_offset, sigma);
  const auto& wide_preproc = at::clamp(wide_normalized, 0., 10.);
  const auto& user_emb_t = at::transpose(user_emb, 1, 2);
  const auto& dp_unflatten = at::bmm(ad_emb_packed, user_emb_t);
  const auto& dp = at::flatten(dp_unflatten, 1, -1);
  const auto& inp = at::cat({dp, wide_preproc}, 1);
  const auto& mm = at::mm(inp, at::t(fc_w));
  const auto& fc1 = at::add(fc_b, mm);
  return at::sigmoid(fc1);
  //const torch::Tensor qq = torch::tensor({{1,2,3}, {2,3,4}});
  //return qq;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("my_fwd", &my_fwd);
}
"""

if False:
    torch.utils.cpp_extension.load_inline(
        name="my_fwd",
        cpp_sources=fwd_op_cpp,
        is_python_module=False,
        extra_cflags=['-O3', '-ffast-math'],
        verbose=True,
    )

    result = torch.ops.my_ops.my_fwd(a, b)


class Once(set):
    def __call__(self, *x):
        if x in self:
            return False
        self.add(x)
        return True


def make_conv(conv):
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    groups = conv.groups
    padding = conv.padding
    dilation = conv.dilation
    kernel_size = conv.kernel_size
    stride = conv.stride
    weight = conv.weight
    bias = conv.bias
    if bias is not None:
        assert len(bias.shape) == 1
        bias = bias.view(1, bias.shape[0], 1, 1)
    assert conv.padding_mode == "zeros"

    def _conv(image):
        batch_size, _, *in_sizes = image.shape
        out_sizes = [
            (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
            for i, v in enumerate(in_sizes)]
        out = torch.zeros([batch_size, out_channels] + out_sizes)

        for n in range(batch_size):
            for group in range(groups):
                for out_chan_part in range(out_channels // groups):
                    for in_chan_part in range(in_channels // groups):
                        for out0 in range(out_sizes[0]):
                            for out1 in range(out_sizes[1]):
                                for kernel0 in range(0, kernel_size[0]):
                                    for kernel1 in range(0, kernel_size[1]):
                                        in_chan = in_chan_part + group * (in_channels // groups)
                                        out_chan = out_chan_part + group * (out_channels // groups)
                                        in0 = out0 * stride[0] + kernel0 * dilation[0] - padding[0]
                                        in1 = out1 * stride[1] + kernel1 * dilation[1] - padding[1]
                                        if 0 <= in0 < in_sizes[0] and 0 <= in1 < in_sizes[1]:
                                            out[n, out_chan, out0, out1] += (
                                                    weight[out_chan, in_chan_part, kernel0, kernel1] *
                                                    image[n, in_chan, in0, in1]
                                            )

        if bias is not None:
            out += bias
        return out

    return _conv


def test_conv(conv: torch.nn.Conv2d, example_input: torch.Tensor):
    correct = conv(example_input)
    result = make_conv(conv)(example_input)

    assert result.shape == correct.shape, f"{result.shape} {correct.shape}"
    torch.testing.assert_allclose(correct, result)
    pass


def unittests():
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 4, (3, 1), stride=1, padding=0), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=1, dilation=2), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=8), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=1, padding=0, groups=2), torch.randn(2, 8, 8, 8))
    test_conv(torch.nn.Conv2d(8, 8, (3, 3), stride=2, padding=5, groups=4, bias=False), torch.randn(2, 8, 16, 8))


def main():
    unittests()

    return
    first = Once()
    testcases = pd.read_csv("testcases.csv").sort_values("gflops")
    for _, row in testcases.iterrows():
        conv_args = literal_eval(row["conv2d"])
        input_shape = literal_eval(row["input"])
        if first(conv_args, input_shape):
            print(conv_args, input_shape)
            test_conv(torch.nn.Conv2d(*conv_args),
                      torch.randn(input_shape))
        if len(first) > 10:
            return


if __name__ == "__main__":
    main()
