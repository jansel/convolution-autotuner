import ctypes
import gc
import logging
import math
import os
from _ctypes import dlclose
from multiprocessing import Process
from subprocess import check_call, Popen, PIPE
from tempfile import NamedTemporaryFile

import sympy
import torch

from convtuner import ir
from convtuner.ir import Assign
from convtuner.ir import Block
from convtuner.ir import Condition
from convtuner.ir import IVDepLoopRange
from convtuner.ir import Literal
from convtuner.ir import Load
from convtuner.ir import LoopRange
from convtuner.ir import Loops
from convtuner.ir import ReduceStore
from convtuner.ir import Store
from convtuner.ir import Sum
from convtuner.ir import VectorStore
from convtuner.ir import expand
from convtuner.ir import is_true
from convtuner.ir import tmpvar
from convtuner.utils import timer

VERBOSE = False
PREFIX_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix.h")
STATIC_WEIGHTS = False
PERMUTE_WEIGHTS = False

log = logging.getLogger(__name__)


def make_convolution(constants, cfg):
    fuse_limit = cfg.integer("fuse_limit", 0, 10)
    use_memset = cfg.boolean("use_memset") and not constants["bias"]

    regions = [
        Loops(reorder_loops(cfg, constants, loops), convolution_body())
        for loops in nine_way_split(convolution_loopranges, constants)
    ]

    if not use_memset:
        prefixes = [
            Loops(reorder_loops(cfg, constants, loops), bias_body(constants))
            for loops in nine_way_split(pointwise_loopranges, constants)
        ]
        regions = [Block([p, r]).fuse_loops(fuse_limit)
                   for p, r in zip(prefixes, regions)]

    code = Block(regions).fuse_loops(fuse_limit)

    if use_memset:
        size = expand("sizeof(float) * batch_size * out_channels * out_sizes0 * out_sizes1").subs(constants)
        code = Block([Literal(f"memset(out, 0, {size});"), code])
    return code.subs(constants)


def convolution_loopranges(out_begin0=0, out_end0="out_sizes0",
                           out_begin1=0, out_end1="out_sizes1"):
    return [LoopRange("n", "batch_size"),
            LoopRange("g", "groups"),
            LoopRange("out_channel_g", "out_channels // groups"),
            LoopRange("in_channel_g", "in_channels // groups"),
            LoopRange("out0", out_begin0, out_end0),
            LoopRange("out1", out_begin1, out_end1),
            LoopRange("kernel0", "kernel_size0"),
            LoopRange("kernel1", "kernel_size1")]


def pointwise_loopranges(out_begin0=0, out_end0="out_sizes0",
                         out_begin1=0, out_end1="out_sizes1"):
    return [LoopRange("n", "batch_size"),
            LoopRange("g", "groups"),
            LoopRange("out_channel_g", "out_channels // groups"),
            LoopRange("out0", out_begin0, out_end0),
            LoopRange("out1", out_begin1, out_end1)]


def convolution_body():
    macros = {"in_channel": expand("in_channel_g + g * (in_channels // groups)"),
              "out_channel": expand("out_channel_g + g * (out_channels // groups)"),
              "in0": expand("out0 * stride0 + kernel0 * dilation0 - padding0"),
              "in1": expand("out1 * stride1 + kernel1 * dilation1 - padding1"), }
    return Condition(["0 <= in0", "in0 < in_sizes0",
                      "0 <= in1", "in1 < in_sizes1"],
                     Sum(
                         Store.from_indices("out", ["n", "out_channel", "out0", "out1"]),
                         [Load.from_indices("weight", ["out_channel", "in_channel_g", "kernel0", "kernel1"]),
                          Load.from_indices("image", ["n", "in_channel", "in0", "in1"])],
                         "v0 * v1")).subs(macros)


def bias_body(constants):
    macros = {"out_channel": expand("out_channel_g + g * (out_channels // groups)")}
    if constants["bias"]:
        return Assign(
            Store.from_indices("out", ["n", "out_channel", "out0", "out1"]),
            [Load.from_indices("bias", ["out_channel"])]
        ).subs(macros)
    else:
        return Assign(
            Store.from_indices("out", ["n", "out_channel", "out0", "out1"]),
            [Literal("0")]
        ).subs(macros)


def funcdef(body, constants):
    if STATIC_WEIGHTS:
        defs = f"""
            const float* __restrict__ weight = (float* __restrict__) 0x{constants["weight_data_ptr"]:x}L;
            const float* __restrict__ bias = (float* __restrict__) 0x{constants.get("bias_data_ptr", 0):x}L;
        """.lstrip()
        args = ""
    else:
        defs = ""
        args = """
            const float* __restrict__ weight,
            const float* __restrict__ bias,
        """.lstrip()
    return f"""
        #include "{PREFIX_HEADER}"
        void convolution(
            const float* __restrict__ image,
            {args}
            float* __restrict__ out
        )
        {{
            {defs}
            {body}
        }}
        """


def reorder_loops(cfg, constants, loops):
    loops = [l.subs(constants) for l in loops]
    order = cfg.permutation("loop_order",
                            ["n", "g",
                             "out0", "out1",
                             "out_channel_g", "in_channel_g",
                             "kernel0", "kernel1"])
    batch_first = cfg.boolean("batch_first")
    if batch_first:
        # bias towards n coming first
        order = ["n"] + [x for x in order if x != "n"]

    name_to_idx = {k: i for i, k in enumerate(loop.name for loop in loops)}
    order = [k for k in order if k in name_to_idx]
    assert len(order) == len(name_to_idx), f"len({order}) != len({name_to_idx})"
    loops = [loops[name_to_idx[k]] for k in order]

    outer = []
    inner = []
    for loop in loops:
        assert loop.step == 1
        tiling = cfg.power_of_two(f"tiling_{loop.name}", 1, 128)
        uneven = cfg.boolean(f"tiling_{loop.name}_uneven")
        width = int(expand(loop.end - loop.begin))
        var = loop.name + "_tile"
        if loop.name == "n" and batch_first:
            tiling = 1

        if tiling >= width:
            inner.append(loop)
        elif tiling > 1 and width % tiling == 0:
            outer.append(LoopRange(var, loop.begin, loop.end, tiling))
            inner.append(LoopRange(loop.name, var, expand(f"{var} + {tiling}")))
        elif tiling >= 8 and uneven:
            outer.append(LoopRange(var, loop.begin, loop.end, tiling))
            inner.append(LoopRange(loop.name, var, expand(f"Min({var} + {tiling}, {loop.end})")))
        else:
            outer.append(loop)

    return outer + inner


def nine_way_split(fn, constants):
    c = constants
    begin0 = int(math.ceil(c["padding0"] / c["stride0"]))
    begin1 = int(math.ceil(c["padding1"] / c["stride1"]))
    end0 = int(math.ceil((c["in_sizes0"] - c["kernel_size0"] + 1 + c["padding0"]) / c["stride0"]))
    end1 = int(math.ceil((c["in_sizes1"] - c["kernel_size1"] + 1 + c["padding1"]) / c["stride1"]))
    assert end0 <= c["out_sizes0"]
    assert end1 <= c["out_sizes1"]
    return [
        fn(out_begin0=0, out_end0=begin0,
           out_begin1=0, out_end1=begin1),
        fn(out_begin0=0, out_end0=begin0,
           out_begin1=begin1, out_end1=end1),
        fn(out_begin0=0, out_end0=begin0,
           out_begin1=end1, out_end1=c["out_sizes1"]),
        fn(out_begin0=begin0, out_end0=end0,
           out_begin1=0, out_end1=begin1),
        fn(out_begin0=begin0, out_end0=end0,
           out_begin1=begin1, out_end1=end1),
        fn(out_begin0=begin0, out_end0=end0,
           out_begin1=end1, out_end1=c["out_sizes1"]),
        fn(out_begin0=end0, out_end0=c["out_sizes0"],
           out_begin1=0, out_end1=begin1),
        fn(out_begin0=end0, out_end0=c["out_sizes0"],
           out_begin1=begin1, out_end1=end1),
        fn(out_begin0=end0, out_end0=c["out_sizes0"],
           out_begin1=end1, out_end1=c["out_sizes1"]),
    ]


def ivdep(node):
    if isinstance(node, Loops):
        memory = node.find_all((Store, VectorStore, ReduceStore))
        ranges = []
        for rng in reversed(node.ranges):
            if all(is_true(sympy.diff(s.index, str(rng.name).replace("_tile", "")) > 0) for s in memory):
                ranges.append(IVDepLoopRange(rng.name, rng.begin, rng.end, rng.step))
            else:
                ranges.append(rng)
        return Loops(reversed(ranges), node.body)
    return node


def gcc_flags(cfg):
    # This is a pretty small subset of the flags from:
    # https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.htm
    # TODO(jansel): expand list of tuned flags

    flags = []
    return flags

    prefetch_ratio = cfg.integer('prefetch_ratio', 1, 20)
    prefetch_latency = cfg.integer('prefetch_latency', 32, 512)
    prefetch_count = cfg.integer('prefetch_count', 1, 10)
    if cfg.boolean("prefetch"):
        flags.extend([
            f"--param=min-insn-to-prefetch-ratio={prefetch_ratio}",
            f"--param=prefetch-min-insn-to-mem-ratio={prefetch_ratio}",
            f"--param=prefetch-latency={prefetch_latency}",
            f"--param=simultaneous-prefetches={prefetch_count}",
        ])
    else:
        flags.append("-fno-prefetch-loop-arrays")

    stalled = cfg.integer("stalled_insns", 0, 64)
    if cfg.boolean("stalled"):
        flags.extend([
            f"-fsched-stalled-insns={stalled}",
            f"-fsched-stalled-insns-dep={stalled}",
        ])

    unroll = cfg.integer("unroll_insns", 1, 4096)
    if cfg.boolean("unroll"):
        flags.extend([
            "-funroll-loops",
            f"--param=max-average-unrolled-insns={unroll}",
            f"--param=max-unroll-times={unroll}",
            f"--param=max-unrolled-insns={unroll}",
        ])

    tiling = cfg.integer("tile_size", 16, 512)
    if cfg.boolean("tiling"):
        flags.extend([
            "-floop-block",
            f"--param=loop-block-tile-size={tiling}",
        ])

    return [x for x in flags if x]


def codegen_and_compile(cfg, constants, output_filename, standalone_times=None):
    ir.tls.prefix = ""
    ir.tls.cfg = cfg
    tmpvar.count = 0
    c = constants
    # TODO(jansel): debug/support dilation
    assert c["dilation0"] == 1 and c["dilation1"] == 1
    # cxx = cfg.enum("cxx", ["clang-11", "gcc-10"])
    # opt = cfg.enum("opt_level", ["3", "2", "fast"])
    cxx = cfg.enum("cxx", ["gcc-10"])
    opt = cfg.enum("opt_level", ["fast"])

    passes = [
        # ("tiling_and_reorder", lambda body: body.tiling_and_reorder()),
        # ("split_loops", lambda body: body.split_loops(constants)),
        ("simplify", lambda body: body.simplify_conditionals(dict(), dict())),
        # ("vectorize", lambda body: body.vectorize_loops()[0]),
        ("ivdep", lambda body: (body.apply_node(ivdep) if ("gcc" in cxx) else body)),
        ("cache_writes", lambda body: body.cache_writes()[0].simplify()),
        ("print", lambda body: print_code(body, constants) or body),
    ]

    code = make_convolution(constants, cfg)
    for pass_name, pass_fn in passes:
        with timer(pass_name):
            code = pass_fn(code)

    with NamedTemporaryFile(suffix=".c") as source:
        source.write(funcdef(code, constants).encode("utf-8"))
        source.flush()
        cmd = [
            cxx, "-o", output_filename, source.name,
            f"-O{opt}", "-march=native",
        ]
        if standalone_times:
            cmd.extend(f"-D{k}={v}" for k, v in [
                ("IMAGE_LEN", c["image_numel"]),
                ("WEIGHT_LEN", c["weight_numel"]),
                ("BIAS_LEN", c["bias_numel"]),
                ("OUT_LEN", c["out_numel"]),
                ("STANDALONE", 1),
                ("TIMES", standalone_times),
            ])
        else:
            cmd.append("-shared")

        if "gcc" in cxx:
            cmd.extend(gcc_flags(cfg))
        else:
            assert "clang" in cxx
            cmd.extend(["-fPIC"])

        VERBOSE and log.info(" ".join(cmd))
        with timer("compile"):
            check_call(cmd)

    ir.tls.cfg = None


def print_code(code, constants):
    VERBOSE and Popen("clang-format", stdin=PIPE).communicate(funcdef(code, constants).encode("utf-8"))


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
        assert dilation == (1, 1)  # TODO(jansel): fix dilation
        assert conv.padding_mode == "zeros"  # TODO(jansel): support other padding

        # These depend on conv + image, plan to make these dynamic in the future
        batch_size, _, *in_sizes = image.shape
        out_sizes = [
            (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
            for i, v in enumerate(in_sizes)]
        in_sizes0, in_sizes1 = in_sizes
        out_sizes0, out_sizes1 = out_sizes
        # TODO(jansel): support conv3d

        out = torch.zeros([batch_size, out_channels,
                           out_sizes0, out_sizes1], dtype=image.dtype)

        # TODO(jansel): make these dynamic
        image_stride0, image_stride1, image_stride2, image_stride3 = image.stride()
        weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
        out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

        image_numel = image.numel()
        weight_numel = weight.numel()
        out_numel = out.numel()

        if conv.bias is not None:
            bias_stride0, = conv.bias.stride()
            bias_data_ptr = conv.bias.data_ptr()
            bias = 1
            bias_numel = conv.bias.numel()
        else:
            bias_numel = 1
            bias = 0

        weight_data_ptr = conv.weight.data_ptr()
        return {k: v for k, v in locals().items() if isinstance(v, int)}

    def __init__(self, cfg, conv, image, standalone_filename=None, standalone_times=None):
        super(ConvolutionGenerator, self).__init__()
        constants = self.get_constants(conv, image)
        if PERMUTE_WEIGHTS:
            shape = list(conv.weight.shape)
            order = cfg.permutation("weight_permute", list(range(len(shape))))
            self.weight = torch.empty([shape[x] for x in order]).permute(
                *[order.index(x) for x in range(len(shape))])
            assert list(self.weight.shape) == shape, f"{order}, {shape}, {self.weight.shape}"
            self.weight.copy_(conv.weight)
            (constants["weight_stride0"],
             constants["weight_stride1"],
             constants["weight_stride2"],
             constants["weight_stride3"]) = self.weight.stride()
        else:
            self.weight = conv.weight

        self.weight_data = ctypes.c_void_p(self.weight.data_ptr())
        self.bias = conv.bias
        if self.bias is not None:
            self.bias_data = ctypes.c_void_p(self.bias.data_ptr())
        else:
            self.bias_data = ctypes.c_void_p(0)
        self.out_shape = (constants["batch_size"],
                          constants["out_channels"],
                          constants["out_sizes0"],
                          constants["out_sizes1"])

        if standalone_filename:
            assert standalone_times
            self.standalone_codegen_and_compile(cfg, constants, standalone_filename, standalone_times)
            self.compiled = None
        else:
            with NamedTemporaryFile(suffix=".so") as shared_object:
                codegen_and_compile(cfg, constants, shared_object.name)
                lib = ctypes.cdll.LoadLibrary(shared_object.name)
                self.compiled = lib.convolution
                self.handle = lib._handle

    def standalone_codegen_and_compile(self, cfg, constants, standalone_filename, standalone_times):
        p = Process(target=codegen_and_compile, args=(cfg, constants, standalone_filename, standalone_times))
        try:
            p.start()
            p.join()
            assert p.exitcode == 0
        except:
            p.kill()
            raise

    def close(self):
        if self.compiled is not None:
            self.compiled = None
            gc.collect()
            dlclose(self.handle)
            self.handle = None
            gc.collect()

    def __call__(self, image):
        out = torch.empty(self.out_shape)
        if STATIC_WEIGHTS:
            self.compiled(ctypes.c_void_p(image.data_ptr()),
                          ctypes.c_void_p(out.data_ptr()))
        else:
            self.compiled(ctypes.c_void_p(image.data_ptr()),
                          self.weight_data,
                          self.bias_data,
                          ctypes.c_void_p(out.data_ptr()))
        return out
