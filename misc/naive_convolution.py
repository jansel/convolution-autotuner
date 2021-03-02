import torch


def naive_convolution(conv):
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
