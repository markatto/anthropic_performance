"""Depth-aware unrolled main loop that eliminates all flow ops."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from loops.common import Instruction, build_hash, build_parity_index

if TYPE_CHECKING:
    from kernel import ScratchLayout


def build_main_loop_unrolled(
    s: "ScratchLayout",
    batch_size: int,
    rounds: int,
    forest_height: int,
    const_fn: Callable[[int], int],
) -> list[Instruction]:
    """Scalar main loop with depth-aware wrap elimination.

    Since all batch indices start at root (depth 0), we can track depth
    statically and eliminate all flow ops:
    - Normal rounds (depth < forest_height): no wrap check needed
    - Wrap rounds (depth == forest_height): force idx = 0 after index computation
    """
    slots: list[Instruction] = []
    depth = 0
    for round in range(rounds):
        is_wrap = (depth == forest_height)
        for batch in range(batch_size):
            cur_node = s.instance_pointers + batch
            acc = s.instance_accumulators + batch

            # Load the value stored at current tree node
            slots.append(("alu", ("+", s.tmp_addr, s.forest_values_p, cur_node)))
            slots.append(("load", ("load", s.tmp_node_val, s.tmp_addr)))
            slots.append(("debug", ("compare", s.tmp_node_val, (round, batch, "node_val"))))

            # XOR with node value and apply hash function
            slots.append(("alu", ("^", acc, acc, s.tmp_node_val)))
            slots.extend(build_hash(acc, s.tmp1, s.tmp2, round, batch, const_fn))
            slots.append(("debug", ("compare", acc, (round, batch, "hashed_val"))))

            # Compute next tree index
            slots.extend(build_parity_index(s, acc, cur_node))
            slots.append(("debug", ("compare", cur_node, (round, batch, "next_idx"))))
            if is_wrap:
                slots.append(("alu", ("+", cur_node, s.zero_const, s.zero_const)))
            slots.append(("debug", ("compare", cur_node, (round, batch, "wrapped_idx"))))

        depth = 0 if is_wrap else depth + 1
    return slots
