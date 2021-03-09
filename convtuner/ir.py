import logging
import re
import threading
from collections import defaultdict, deque
from functools import partial, cmp_to_key, lru_cache, reduce
from itertools import chain
from typing import List

import sympy

VECTOR_SIZE = 8
VECTOR_BITS = VECTOR_SIZE * 32
THREETILE = False
expand = sympy.expand
simplify = partial(sympy.simplify, evaluate=True)
tls = threading.local()
log = logging.getLogger(__name__)


@cmp_to_key
def cmp_expr(a, b):
    if is_true(simplify(a < b)):
        return -1
    elif is_true(simplify(a > b)):
        return 1
    else:
        assert str(a) == str(b)
        return 0


@lru_cache(128)
def is_true(x):
    return isinstance(simplify(nonnegative(x)), sympy.logic.boolalg.BooleanTrue)


@lru_cache(128)
def is_false(x):
    return isinstance(simplify(nonnegative(x)), sympy.logic.boolalg.BooleanFalse)


@lru_cache(128)
def is_boolean(x):
    return isinstance(simplify(nonnegative(x)),
                      (sympy.logic.boolalg.BooleanFalse, sympy.logic.boolalg.BooleanTrue))


def identity(x):
    return x


def nonnegative(x):
    return x.subs({v: sympy.Symbol(v, nonnegative=True, integer=True) for v in map(str, x.free_symbols)})


def tmpvar():
    n = f"tmp{tmpvar.count}"
    tmpvar.count += 1
    return n


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

    def __repr__(self):
        return f"""{self.__class__.__name__}({','.join(map('"{}"'.format, [
            self.name, self.begin, self.end, self.step
        ]))})"""

    def __init__(self, name, begin, end=None, step=1):
        super(LoopRange, self).__init__()
        self.name = name
        if end is None:  # mimic range()
            begin, end = 0, begin
        self.begin = expand(begin)
        self.end = expand(end)
        self.step = expand(step)

    @property
    def width(self):
        return int(expand((self.end - self.begin) // self.step))

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(name=self.name,
                                      begin=expr_fn(self.begin),
                                      end=expr_fn(self.end),
                                      step=expr_fn(self.step)))

    def can_vectorize(self):
        return (is_true(self.begin + VECTOR_SIZE <= self.end) and
                expand(((self.end - self.begin) // self.step) % VECTOR_SIZE) == 0)

    def tile(self, tiling):
        width = expand(self.end - self.begin)
        assert self.step == 1  # TODO(jansel): support this
        assert width.is_integer  # TODO(jansel): support this
        width = int(width)
        var = str(self.name)

        max_tiling = 1
        while width > 0 and (width % (max_tiling * 2)) == 0:
            max_tiling *= 2

        if tiling >= width:
            tiling = width
        elif tiling > max_tiling:
            tiling = max_tiling

        t0 = f"{var}_t0"
        t1 = f"{var}_t1"

        assert width % tiling == 0
        return (LoopRange(t0, width // tiling),
                LoopRange(t1, tiling),
                {var: expand(f"{t0} * {tiling} + {t1} + {self.begin}")})


class IVDepLoopRange(LoopRange):
    def __str__(self):
        return "#pragma GCC ivdep\n" + super(IVDepLoopRange, self).__str__()


class Loops(Node):
    def __str__(self):
        return "\n".join(map(str, self.ranges + [self.body]))

    def __init__(self, ranges, body):
        super(Loops, self).__init__()
        self.ranges: List[LoopRange] = list(ranges)
        self.body: Node = body

    def apply(self, node_fn, expr_fn):
        if self.body is None:
            return None
        body = self.body.apply(node_fn, expr_fn)
        if body is None:
            return None
        new_ranges = []
        for r in self.ranges:
            r = r.apply(node_fn, expr_fn)
            if (is_false(expand(r.begin + r.step < r.end)) and
                    is_true(expand(r.begin < r.end))):
                body = body.subs({r.name: r.begin})
            else:
                new_ranges.append(r)
        if len(self.ranges) == 0:
            return node_fn(body)
        return node_fn(self.__class__(new_ranges, body))

    def simplify_conditionals(self, first, last):
        if any(is_false(r.begin < r.end) for r in self.ranges) or self.body is None:
            return None
        ranges = list(self.ranges)
        body = self.body

        if isinstance(body, Condition):
            for test in body.tests:
                if len(test.free_symbols) == 1:
                    var = str(list(test.free_symbols)[0])
                    for i in range(len(ranges)):
                        if ranges[i].name == var and ranges[i].step == 1:
                            # Narrow this loop range to remove the conditional
                            begin = ranges[i].begin
                            end = ranges[i].end
                            step = ranges[i].step
                            if is_false(test.subs({var: begin})) and is_false(test.subs({var: end - step})):
                                break  # unreachable code
                            while is_false(test.subs({var: begin})) and is_true(begin < end):
                                begin = begin + step
                            while is_false(test.subs({var: end - step})) and is_true(begin < end):
                                end = end - step
                            ranges[i] = LoopRange(var, begin, end, step)

        first = dict(first)
        last = dict(last)
        for rng in ranges:
            first[rng.name] = expand(rng.begin.subs(first))
            last[rng.name] = expand((rng.end - 1).subs(last))
        return Loops(ranges, body.simplify_conditionals(first, last))

    def insert_prefix_suffix(self, depth, prefix, suffix):
        upper_ranges = self.ranges[:depth]
        lower_ranges = self.ranges[depth:]
        assert upper_ranges and lower_ranges
        return Loops(lower_ranges, Block([
            prefix,
            Loops(upper_ranges, self.body),
            suffix
        ]))

    def cache_writes(self):
        if tls.cfg.boolean(f"{tls.prefix}skip_cache_writes"):
            return self, []

        body, stores = self.body.cache_writes()
        stores = set(stores)
        ranges = []
        seen = []
        defs = []
        sums = []

        def do_caching(store):
            nonlocal defs, sums, body, seen

            def matches(n):
                return (store.name == n.name and
                        is_true(sympy.Equality(store.index, n.index)))

            if any(map(matches, seen)):
                return
            seen.append(store)

            var = tmpvar()
            if isinstance(store, (ReduceStore, VectorStore)):
                defs.append(TempDef(var).vectorize())
            else:
                defs.append(TempDef(var))

            sums.append(Sum(store, [TempRef(var)], "v0"))

            def swap(n):
                if isinstance(n, (Store, ReduceStore, VectorStore)) and matches(n):
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
        if tls.cfg.boolean(f"{tls.prefix}skip_vectorize"):
            return self, [], set()
        body, memory, banned = self.body.vectorize_loops()
        if any(m is False for m in memory):
            return Loops(self.ranges, body), [False], banned
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

        result = Loops(reversed(ranges), body)
        if not first:
            memory = [False]  # stop
            # print(f"BEFORE\n{node}\nAFTER\n{result}")
        return result, memory, banned

    def tiling_and_reorder(self):
        ranges = defaultdict(list)
        body = self.body
        for r in self.ranges:
            width = expand(r.end - r.begin)
            assert r.step == 1  # TODO(jansel): support this
            assert width.is_integer  # TODO(jansel): support this
            width = int(width)
            var = str(r.name)

            lg_max_tiling = 0
            max_tiling = 1
            while width > 0 and (width % (max_tiling * 2)) == 0:
                max_tiling *= 2
                lg_max_tiling += 1

            tiling = tls.cfg.float(f"{tls.prefix}tiling_{var}", 0.0, 1.0)
            if THREETILE:
                tiling2 = tls.cfg.float(f"{tls.prefix}tiling2_{var}", 0.0, 1.0)

            if max_tiling > 1:
                t0 = f"{var}_t0"
                t1 = f"{var}_t1"
                tiling = int(round(lg_max_tiling * tiling))
                tiling = 2 ** tiling
                if THREETILE:
                    t2 = f"{var}_t2"
                    tiling2 = int(round((lg_max_tiling - tiling) * tiling2))
                    tiling2 = 2 ** tiling2
                    assert width % (tiling * tiling2) == 0
                    ranges[var] = [LoopRange(t0, width // (tiling2 * tiling)),
                                   LoopRange(t2, tiling2),
                                   LoopRange(t1, tiling)]
                    body = body.subs({var: expand(
                        f"{t0} * {tiling2 * tiling} + {t2} * {tiling} + {t1} + {r.begin}")})
                else:
                    assert width % (tiling) == 0

                    ranges[var] = [LoopRange(t0, width // tiling),
                                   LoopRange(t1, tiling)]
                    body = body.subs({var: expand(
                        f"{t0} * {tiling} + {t1} + {r.begin}")})
            else:
                ranges[var] = [r]

        options = ["n", "g", "out_channel_g", "in_channel_g",
                   "out0", "out1", "kernel0", "kernel1"] * (2 + int(THREETILE))
        loop_order = tls.cfg.permutation(
            f"{tls.prefix}loop_order", list(reversed((options))))
        assert len(loop_order) == len(options), f"{loop_order} != {options}"

        new_ranges = []
        for var in loop_order:
            if ranges[var]:
                new_ranges.append(ranges[var].pop())

        assert all(len(x) == 0 for x in ranges.values())

        return Loops(reversed(new_ranges), body)

    def split_loops(self, constants):
        limit = tls.cfg.integer(f"{tls.prefix}split_loops_limit", 0, 2)  # exponential;
        threshold = tls.cfg.integer(f"{tls.prefix}split_loops_threshold", 1, 128)
        conds = list(chain(*[c.tests for c in self.find_all(Condition)]))
        if limit <= 0 or len(conds) == 0:
            return self

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
                        if not splits or is_true(splits[0] < idx):
                            splits.append(idx)
                        break
            if splits:
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
                limit -= 1
                continue
            ranges.insert(0, rng)
        return Loops(ranges, body)

    def unroll(self, limit):
        ranges = list(self.ranges)
        body = [self.body]
        while ranges and ranges[-1].width * len(body) <= limit:
            new_body = []
            rng = ranges.pop()
            for i in range(rng.width):
                new_body.extend([
                    b.subs({rng.name: rng.begin + rng.step * i})
                    for b in body])
            body = new_body
        return Loops(ranges, Block(body))

    def fuse_pointwise_prefix(self, pointwise: "Loops"):
        input_ranges = deque(self.ranges)
        output_ranges = []

        conditions = []
        for rng1 in pointwise.ranges:
            rng2 = input_ranges.popleft()
            while str(rng1) != str(rng2):
                conditions.append(sympy.Eq(sympy.Symbol(rng2.name), rng2.begin))
                output_ranges.append(rng2)
                rng2 = input_ranges.popleft()
            output_ranges.append(rng2)

        return Loops(output_ranges,
                     Block([
                         Condition(conditions, pointwise.body),
                         Loops(list(input_ranges), self.body)
                     ]))


class Condition(Node):
    def __str__(self):
        def _str(s):
            if isinstance(s, sympy.Eq):
                return f"{s.lhs} == {s.rhs}"
            return str(s)

        test = " && ".join(map(_str, self.tests))
        return f"if({test})\n{self.body}"

    def __init__(self, tests, body):
        super(Condition, self).__init__()
        self.tests: List[sympy.Expr] = list(map(expand, tests))
        self.body: Node = body

    def apply(self, node_fn, expr_fn):
        tests = [expr_fn(x) for x in self.tests]
        tests = [x for x in tests if not is_true(x)]
        body = self.body.apply(node_fn, expr_fn)
        if not tests:
            return body
        if any(map(is_false, tests)):
            return None
        return node_fn(self.__class__(tests, body))

    def simplify_conditionals(self, first_sub, last_sub):
        tests = []
        for t in self.tests:
            # this assumes conditionals are monotonic
            first = t.subs(first_sub)
            last = t.subs(last_sub)
            if is_true(first) and is_true(last):
                tests.append(expand(True))
            elif is_false(first) and is_false(last):
                tests.append(expand(False))
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


class Statement(Node):
    def __init__(self, output, inputs, expression="v0"):
        super(Statement, self).__init__()
        self.output: Memory = output
        self.inputs: List[Memory] = list(inputs)
        self.expression: sympy.Expr = expand(expression)

    def apply(self, node_fn, expr_fn):
        output = self.output.apply(node_fn, expr_fn)
        inputs = [x.apply(node_fn, expr_fn) for x in self.inputs]
        expression = expr_fn(self.expression)
        return node_fn(self.__class__(output, inputs, expression))

    def cache_writes(self):
        return self, self.find_all((Store, ReduceStore, VectorStore))

    def vectorize_loops(self):
        return self, self.find_all(Memory), set()


class Reduction(Statement):
    pass


class Sum(Reduction):
    def __str__(self):
        expr = str(self.expression)
        for i, v in enumerate(self.inputs):
            expr = re.sub(fr"\bv{i}\b", str(v), expr)
        if isinstance(self.output, ReduceStore):
            return f"{self.output} += _mm{VECTOR_BITS}_reduce_add_ps({expr});"
        if isinstance(self.output, VectorStore):
            if str(self.expression) == "v0*v1":
                return self.output.fma_cpp(*self.inputs)
            else:
                assert str(self.expression) == "v0"
                return self.output.add_cpp(*self.inputs)
        return f"{self.output} += {expr};"


class Assign(Reduction):
    def __str__(self):
        expr = str(self.expression)
        for i, v in enumerate(self.inputs):
            expr = re.sub(fr"\bv{i}\b", str(v), expr)
        if expr == "0" and isinstance(self.output, VectorizedMemory):
            expr = "_mm256_setzero_ps()"
        return f"{self.output} = {expr};"


class Block(Node):
    def __init__(self, statements):
        super(Block, self).__init__()
        self.statements = [x for x in statements if x is not None]

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
        return node_fn(self.__class__(stmts))

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
            if a is not None:
                statements.append(a)
            if b is not None:
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

    def fuse_loops(self, limit):
        if not all(isinstance(s, Loops) for s in self.statements):
            return self

        lifted = []
        not_lifted = [list(s.ranges) for s in self.statements]
        bodies = [s.body for s in self.statements]

        while len(lifted) < limit and all(not_lifted) and len(set(str(l[0]) for l in not_lifted)) == 1:
            # all loops the same
            lifted.append(not_lifted[0][0])
            for r in not_lifted:
                r.pop(0)

        code = Block([Loops(r, b) for r, b in zip(not_lifted, bodies)])
        if lifted:
            code = Loops(lifted, code)
        return code

    def unroll(self, limit):
        return Block([x.unroll(limit) for x in self.statements])


class Memory(Node):
    @classmethod
    def from_indices(cls, name, indices):
        assert isinstance(indices, list)
        return cls(name, sum(sympy.Mul(expand(v), expand(f"{name}_stride{i}"))
                             for i, v in enumerate(indices)))

    def __init__(self, name, index):
        super(Memory, self).__init__()
        self.name = name
        self.index = expand(index)

    def __str__(self):
        return f"{self.name}[{self.index}]"

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(self.name, expr_fn(self.index)))


class Literal(Node):
    def __init__(self, value):
        super(Literal, self).__init__()
        self.value = value

    def __str__(self):
        return self.value

    def apply(self, node_fn, expr_fn):
        return node_fn(self.__class__(self.value))

    def vectorize_loops(self):
        return self, [], set()

    def cache_writes(self):
        return self, []


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
    def __str__(self):
        # TODO(jansel): generate aligned loads
        return f"_mm256_loadu_ps({self.name} + {self.index})"


class VectorStore(VectorizedMemory):
    def fma_cpp(self, a, b):
        # TODO(jansel): generate aligned loads
        return f"fma_storeu({self.name} + {self.index}, {a}, {b});"

    def add_cpp(self, a):
        # TODO(jansel): generate aligned loads
        return f"add_storeu({self.name} + {self.index}, {a});"


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
