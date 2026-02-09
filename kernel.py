"""
Kernel implementation for the hash-based tree traversal problem.

=============================================================================
ALGORITHM OVERVIEW
=============================================================================

This kernel performs batched tree traversal with hash-based branching decisions.
It processes multiple independent traversals in parallel over several rounds.

Data Structures:
  - Tree: A perfect balanced binary tree (height=10, 2047 nodes) stored as an
    array. Node i has children at indices 2*i+1 (left) and 2*i+2 (right).
  - Batch: 256 independent traversals, each with:
      - indices[i]: current position in the tree
      - values[i]: accumulator value that determines branching

Algorithm (per round, per batch element):
  1. Load current index and value from memory
  2. Load node value from tree at current index
  3. XOR value with node value
  4. Apply hash function (6 stages of mixing operations)
  5. Compute next index: 2*idx + (1 if hash is even, else 2)
  6. Wrap index to 0 if it exceeds tree bounds
  7. Store updated index and value back to memory

The hash function applies 6 stages, each computing:
    tmp1 = val op1 const1
    tmp2 = val op3 const3
    val  = tmp1 op2 tmp2

Test parameters: forest_height=10, rounds=16, batch_size=256

=============================================================================
TARGET ARCHITECTURE
=============================================================================

Custom VLIW SIMD machine with the following execution units per cycle:
  - alu:   12 slots - scalar arithmetic (+, -, *, //, %, ^, &, |, <<, >>, <, ==)
  - valu:   6 slots - vector arithmetic (VLEN=8 elements per vector)
  - load:   2 slots - memory reads and constant loads
  - store:  2 slots - memory writes
  - flow:   1 slot  - control flow (select, jumps, pause)

Key characteristics:
  - All instruction effects apply at end of cycle (no RAW hazards within cycle)
  - Scratch space: 1536 words (serves as registers + manually-managed cache)
  - SIMD width: 8 elements (VLEN)
  - 32-bit word size

=============================================================================
MEMORY LAYOUT
=============================================================================

Header (addresses 0-6):
  [0] rounds          - number of traversal rounds
  [1] n_nodes         - total nodes in tree (2^(height+1) - 1)
  [2] batch_size      - number of parallel traversals
  [3] forest_height   - tree height
  [4] forest_values_p - pointer to tree node values array
  [5] inp_indices_p   - pointer to batch indices array
  [6] inp_values_p    - pointer to batch values array

Data sections:
  [forest_values_p ... inp_indices_p)  - tree node values
  [inp_indices_p   ... inp_values_p)   - batch indices (read/write)
  [inp_values_p    ... end)            - batch values (read/write)

=============================================================================
"""

from problem import VLEN

from loops import build_main_loop_unrolled

Slot = tuple
Instruction = tuple[str, Slot]  # ("engine", (op, ...))
Bundle = dict[str, list[Slot]]  # {"engine": [(op, ...), ...], ...}


HEADER_SIZE = 7


class ScratchLayout:
    """Scratch space addresses allocated during kernel setup.

    All header values are known at build time, so we use scratch_const
    instead of loading from memory.
    """
    def __init__(self, kb, n_nodes: int, batch_size: int):
        alloc = kb.alloc_scratch
        # temporaries (reused across iterations)
        self.tmp1 = alloc("tmp1")
        self.tmp2 = alloc("tmp2")
        self.tmp3 = alloc("tmp3")
        self.tmp_node_val = alloc("tmp_node_val")
        self.tmp_addr = alloc("tmp_addr")
        # memory layout pointers (precomputed from known structure)
        self.forest_values_p = kb.scratch_const(HEADER_SIZE)
        self.inp_indices_p = kb.scratch_const(HEADER_SIZE + n_nodes)
        self.inp_values_p = kb.scratch_const(HEADER_SIZE + n_nodes + batch_size)
        self.n_nodes = kb.scratch_const(n_nodes)
        # small constants
        self.zero_const = kb.scratch_const(0)
        self.one_const = kb.scratch_const(1)
        self.two_const = kb.scratch_const(2)

        kb.instrs.append({"flow": [("pause",)]})

        # per-batch arrays
        self.instance_pointers = alloc("instance_pointers", batch_size)
        self.instance_accumulators = alloc("instance_accumulators", batch_size)


def build_batch_preload(
    kb, batch_size: int, s: ScratchLayout,
) -> list[Bundle]:
    """Load batch indices and accumulators from memory into scratch via vload.

    Uses 2 running address registers, loading both arrays in parallel
    (2 vloads per cycle). Reads and increments happen in the same bundle
    since reads occur before writes commit.
    """

    #TODO: s.inp_indices_p / s.inp_values_p directly for the first vload?
    assert batch_size % VLEN == 0
    vlen_const = kb.scratch_const(VLEN)
    addr_i, addr_v = s.tmp1, s.tmp2

    instrs: list[Bundle] = [
        # Copy base pointers into running address registers
        {"alu": [("+", addr_i, s.inp_indices_p, s.zero_const),
                 ("+", addr_v, s.inp_values_p, s.zero_const)]},
    ]
    for i in range(batch_size // VLEN):
        off = i * VLEN
        instrs.append({
            "load": [("vload", s.instance_pointers + off, addr_i),
                     ("vload", s.instance_accumulators + off, addr_v)],
            "alu": [("+", addr_i, addr_i, vlen_const),
                    ("+", addr_v, addr_v, vlen_const)],
        })
    return instrs


def build_batch_store(
    kb, batch_size: int, s: ScratchLayout,
) -> list[Bundle]:
    """Store batch accumulators back to memory via vstore.

    Uses 2 interleaved address registers to fill both store slots per cycle.
    """
    assert batch_size % (VLEN * 2) == 0
    vlen2_const = kb.scratch_const(VLEN * 2)
    vlen_const = kb.scratch_const(VLEN)
    addr1, addr2 = s.tmp1, s.tmp2

    instrs: list[Bundle] = [
        # addr1 = base, addr2 = base + VLEN
        {"alu": [("+", addr1, s.inp_values_p, s.zero_const),
                 ("+", addr2, s.inp_values_p, vlen_const)]},
    ]
    for i in range(batch_size // VLEN // 2):
        off = i * VLEN * 2
        instrs.append({
            "store": [("vstore", addr1, s.instance_accumulators + off),
                      ("vstore", addr2, s.instance_accumulators + off + VLEN)],
            "alu": [("+", addr1, addr1, vlen2_const),
                    ("+", addr2, addr2, vlen2_const)],
        })
    return instrs


def build_kernel(kb, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    """
    Build the full kernel: preload, main loop, store.

    Args:
        kb: KernelBuilder instance (provides alloc_scratch, add, scratch_const, etc.)
        forest_height: height of the tree
        n_nodes: number of nodes in the tree
        batch_size: number of elements to process in parallel
        rounds: number of iterations
    """
    s = ScratchLayout(kb, n_nodes, batch_size)

    kb.instrs.extend(build_batch_preload(kb, batch_size, s))

    loop_slots = build_main_loop_unrolled(
        s, batch_size, rounds, forest_height, kb.scratch_const,
    )
    kb.instrs.extend(kb.build(loop_slots))

    kb.instrs.extend(build_batch_store(kb, batch_size, s))
    kb.instrs.append({"flow": [("pause",)]})
