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

from typing import Callable

from problem import HASH_STAGES

Slot = tuple
Instruction = tuple[str, Slot]  # ("engine", (op, ...))
Bundle = dict[str, list[Slot]]  # {"engine": [(op, ...), ...], ...}


HEADER_FIELDS = [
    "rounds",          # mem[0]: number of traversal rounds
    "n_nodes",         # mem[1]: total nodes in tree
    "batch_size",      # mem[2]: number of parallel traversals
    "forest_height",   # mem[3]: tree height
    "forest_values_p", # mem[4]: pointer to tree node values
    "inp_indices_p",   # mem[5]: pointer to current indices for each traversal
    "inp_values_p",    # mem[6]: pointer to current values for each traversal
]


class ScratchLayout:
    """Scratch space addresses allocated during kernel setup.

    Allocates all scratch registers and emits header-load instructions into kb.
    """
    def __init__(self, kb, batch_size: int):
        alloc = kb.alloc_scratch
        # temporaries (reused across iterations)
        self.tmp1 = alloc("tmp1")
        self.tmp2 = alloc("tmp2")
        self.tmp3 = alloc("tmp3")

        # load the memory header into named scratch slots
        for i, name in enumerate(HEADER_FIELDS):
            alloc(name, 1)
            kb.add("load", ("const", self.tmp1, i))
            kb.add("load", ("load", kb.scratch[name], self.tmp1))

        kb.add("flow", ("pause",))
        kb.add("debug", ("comment", "Starting loop"))

        self.tmp_node_val = alloc("tmp_node_val")
        self.tmp_addr = alloc("tmp_addr")
        # constants
        self.zero_const = kb.scratch_const(0)
        self.one_const = kb.scratch_const(1)
        self.two_const = kb.scratch_const(2)
        # per-batch arrays
        self.instance_pointers = alloc("instance_pointers", batch_size)
        self.instance_accumulators = alloc("instance_accumulators", batch_size)


def build_hash(
    val_addr: int,
    tmp1: int,
    tmp2: int,
    round: int,
    batch: int,
    const_fn: Callable[[int], int],
) -> list[Instruction]:
    """
    Generate instructions for the 6-stage hash function.

    Each stage computes:
        tmp1 = val op1 const1    (e.g., val + 0x7ED55D16)
        tmp2 = val op3 const3    (e.g., val << 12)
        val  = tmp1 op2 tmp2     (e.g., tmp1 + tmp2)

    The operations mix addition, XOR, and bit shifts to thoroughly
    scramble the input value. See HASH_STAGES in problem.py for the
    specific constants and operations used.
    """
    slots: list[Instruction] = []
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        slots.append(("alu", (op1, tmp1, val_addr, const_fn(val1))))
        slots.append(("alu", (op3, tmp2, val_addr, const_fn(val3))))
        slots.append(("alu", (op2, val_addr, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_addr, (round, batch, "hash_stage", hi))))
    return slots


def build_batch_preload(
    kb, batch_size: int, s: ScratchLayout,
) -> list[Bundle]:
    """Load batch indices and accumulators from memory into scratch."""
    #TODO vload
    instrs: list[Bundle] = []
    for batch in range(batch_size):
        batch_const = kb.scratch_const(batch)
        instrs.append({"alu": [("+", s.tmp_addr, kb.scratch["inp_indices_p"], batch_const)]})
        instrs.append({"load": [("load", s.instance_pointers + batch, s.tmp_addr)]})
        instrs.append({"alu": [("+", s.tmp_addr, kb.scratch["inp_values_p"], batch_const)]})
        instrs.append({"load": [("load", s.instance_accumulators + batch, s.tmp_addr)]})
    return instrs


def build_batch_store(
    kb, batch_size: int, s: ScratchLayout,
) -> list[Bundle]:
    """Store batch accumulators back to memory."""
    #TODO vstore
    instrs: list[Bundle] = []
    for batch in range(batch_size):
        batch_const = kb.scratch_const(batch)
        instrs.append({"alu": [("+", s.tmp_addr, kb.scratch["inp_values_p"], batch_const)]})
        instrs.append({"store": [("store", s.tmp_addr, s.instance_accumulators + batch)]})
    return instrs


def build_kernel(kb, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    """
    Like reference_kernel2 but building actual instructions.
    Scalar implementation using only scalar ALU and load/store.
    
    Args:
        kb: KernelBuilder instance (provides alloc_scratch, add, scratch_const, etc.)
        forest_height: height of the tree
        n_nodes: number of nodes in the tree
        batch_size: number of elements to process in parallel
        rounds: number of iterations
    """
    s = ScratchLayout(kb, batch_size)

    # =========================================================================
    # PRELOAD: Load batch state from memory into scratch
    # =========================================================================
    kb.instrs.extend(build_batch_preload(kb, batch_size, s))

    # =========================================================================
    # MAIN LOOP: Fully unrolled rounds * batch_size iterations
    # =========================================================================
    # The main loop is still built as flat slots and packed via kb.build().
    # This will get its own packing strategy later.
    loop_slots: list[Instruction] = []
    for round in range(rounds):
        for batch in range(batch_size):
            cur_node = s.instance_pointers + batch
            acc = s.instance_accumulators + batch

            # Load the value stored at current tree node
            loop_slots.append(("alu", ("+", s.tmp_addr, kb.scratch["forest_values_p"], cur_node)))
            loop_slots.append(("load", ("load", s.tmp_node_val, s.tmp_addr)))
            loop_slots.append(("debug", ("compare", s.tmp_node_val, (round, batch, "node_val"))))

            # XOR with node value and apply hash function
            loop_slots.append(("alu", ("^", acc, acc, s.tmp_node_val)))
            loop_slots.extend(build_hash(acc, s.tmp1, s.tmp2, round, batch, kb.scratch_const))
            loop_slots.append(("debug", ("compare", acc, (round, batch, "hashed_val"))))

            # Compute next tree index based on hash parity
            loop_slots.append(("alu", ("%", s.tmp1, acc, s.two_const)))
            loop_slots.append(("alu", ("+", s.tmp3, s.tmp1, s.one_const)))
            loop_slots.append(("alu", ("*", cur_node, cur_node, s.two_const)))
            loop_slots.append(("alu", ("+", cur_node, cur_node, s.tmp3)))
            loop_slots.append(("debug", ("compare", cur_node, (round, batch, "next_idx"))))

            # Wrap index back to root if we fell off the tree
            loop_slots.append(("alu", ("<", s.tmp1, cur_node, kb.scratch["n_nodes"])))
            loop_slots.append(("flow", ("select", cur_node, s.tmp1, cur_node, s.zero_const)))
            loop_slots.append(("debug", ("compare", cur_node, (round, batch, "wrapped_idx"))))

    kb.instrs.extend(kb.build(loop_slots))

    # =========================================================================
    # STORE: Write batch accumulators back to memory
    # =========================================================================
    kb.instrs.extend(build_batch_store(kb, batch_size, s))

    kb.instrs.append({"flow": [("pause",)]})
