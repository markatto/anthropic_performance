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

from problem import HASH_STAGES


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
    # =========================================================================
    # SCRATCH ALLOCATION
    # =========================================================================
    # Allocate temporary registers for intermediate computations.
    # These are reused across all iterations of the loop.
    tmp1 = kb.alloc_scratch("tmp1")
    tmp2 = kb.alloc_scratch("tmp2")
    tmp3 = kb.alloc_scratch("tmp3")

    # =========================================================================
    # LOAD MEMORY LAYOUT PARAMETERS FROM HEADER
    # =========================================================================
    # The memory image starts with a 7-word header containing pointers and sizes.
    # We load these into scratch space so we can use them throughout the kernel.
    # See MEMORY LAYOUT in module docstring for details.
    init_vars = [
        "rounds",          # mem[0]: number of traversal rounds
        "n_nodes",         # mem[1]: total nodes in tree
        "batch_size",      # mem[2]: number of parallel traversals
        "forest_height",   # mem[3]: tree height
        "forest_values_p", # mem[4]: pointer to tree node values
        "inp_indices_p",   # mem[5]: pointer to current indices for each traversal
        "inp_values_p",    # mem[6]: pointer to current values for each traversal
    ]
    for v in init_vars:
        kb.alloc_scratch(v, 1)
    for i, v in enumerate(init_vars):
        kb.add("load", ("const", tmp1, i))
        kb.add("load", ("load", kb.scratch[v], tmp1))

    # Pre-load commonly used constants into scratch space
    zero_const = kb.scratch_const(0)
    one_const = kb.scratch_const(1)
    two_const = kb.scratch_const(2)

    # Pause for debugging - matches yield in reference_kernel2
    kb.add("flow", ("pause",))
    kb.add("debug", ("comment", "Starting loop"))

    # =========================================================================
    # MAIN LOOP BODY
    # =========================================================================
    # We build up the loop body as a list of (engine, slot) tuples, then convert
    # to instruction bundles at the end. Currently this is fully unrolled:
    # rounds * batch_size iterations with no actual loop instructions.
    body = []

    # Scratch registers for the per-element computation
    tmp_idx = kb.alloc_scratch("tmp_idx")       # current tree index for this element
    tmp_val = kb.alloc_scratch("tmp_val")       # current hash value for this element
    tmp_node_val = kb.alloc_scratch("tmp_node_val")  # value of current tree node
    tmp_addr = kb.alloc_scratch("tmp_addr")     # computed memory address for loads/stores

    def build_hash(val_hash_addr, tmp1, tmp2, round, i):
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
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, kb.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, kb.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    for round in range(rounds):
        for i in range(batch_size):
            i_const = kb.scratch_const(i)

            # -----------------------------------------------------------------
            # STEP 1: Load current index and value for this batch element
            # -----------------------------------------------------------------
            # idx = mem[inp_indices_p + i]
            # This tells us where we currently are in the tree
            body.append(("alu", ("+", tmp_addr, kb.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("load", tmp_idx, tmp_addr))) # TODO: this can be in scratch
            body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))

            # val = mem[inp_values_p + i]
            # This is our running hash value that determines branching
            body.append(("alu", ("+", tmp_addr, kb.scratch["inp_values_p"], i_const)))
            body.append(("load", ("load", tmp_val, tmp_addr))) # TODO: this can be in scratch
            body.append(("debug", ("compare", tmp_val, (round, i, "val"))))

            # -----------------------------------------------------------------
            # STEP 2: Load the value stored at current tree node
            # -----------------------------------------------------------------
            # node_val = mem[forest_values_p + idx]
            # The tree is stored as a flat array; idx is the node index
            body.append(("alu", ("+", tmp_addr, kb.scratch["forest_values_p"], tmp_idx)))
            body.append(("load", ("load", tmp_node_val, tmp_addr))) # this load is (mostly) necessary
            body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))

            # -----------------------------------------------------------------
            # STEP 3: XOR with node value and apply hash function
            # -----------------------------------------------------------------
            # val = myhash(val ^ node_val)
            # XOR mixes in the tree node's value, then hash scrambles it
            body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
            body.extend(build_hash(tmp_val, tmp1, tmp2, round, i))
            body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))

            # -----------------------------------------------------------------
            # STEP 4: Compute next tree index based on hash parity
            # -----------------------------------------------------------------
            # idx = 2*idx + (1 if val % 2 == 0 else 2)
            # If hash is even, go to left child (2*idx + 1)
            # If hash is odd, go to right child (2*idx + 2)
            body.append(("alu", ("%", tmp1, tmp_val, two_const)))      # tmp1 = val % 2
            body.append(("alu", ("==", tmp1, tmp1, zero_const)))       # tmp1 = (val % 2 == 0)
            # TODO: alu can do this?
            body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))  # tmp3 = 1 or 2
            # TODO: FMA?
            body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))   # idx = 2 * idx
            body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))        # idx = 2*idx + tmp3
            body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))

            # -----------------------------------------------------------------
            # STEP 5: Wrap index back to root if we fell off the tree
            # -----------------------------------------------------------------
            # idx = 0 if idx >= n_nodes else idx
            # If we've gone past the leaves, wrap back to root
            #TODO: this could be a single mod, or fully unrolled or something
            body.append(("alu", ("<", tmp1, tmp_idx, kb.scratch["n_nodes"])))  # tmp1 = idx < n_nodes
            body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))  # idx = idx or 0
            body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))

            # -----------------------------------------------------------------
            # STEP 6: Store updated index and value back to memory
            # -----------------------------------------------------------------
            # TODO: keep this in scratch
            # mem[inp_indices_p + i] = idx
            body.append(("alu", ("+", tmp_addr, kb.scratch["inp_indices_p"], i_const)))
            body.append(("store", ("store", tmp_addr, tmp_idx)))

            # mem[inp_values_p + i] = val
            body.append(("alu", ("+", tmp_addr, kb.scratch["inp_values_p"], i_const)))
            body.append(("store", ("store", tmp_addr, tmp_val)))

    # =========================================================================
    # FINALIZE: Convert slot list to instruction bundles
    # =========================================================================
    # kb.build() takes our list of (engine, slot) tuples and packages them
    # into instruction bundles. Currently each slot becomes its own bundle
    # (no parallelism), but this could be optimized to pack multiple
    # independent operations into the same cycle.
    body_instrs = kb.build(body)
    kb.instrs.extend(body_instrs)

    # Final pause for debugging - matches the final yield in reference_kernel2
    kb.instrs.append({"flow": [("pause",)]})
