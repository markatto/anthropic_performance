"""Baseline scalar main loop with runtime wrap checking."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from loops.common import Instruction, build_hash, build_parity_index

if TYPE_CHECKING:
    from kernel import ScratchLayout


def build_main_loop_scalar(
    s: "ScratchLayout",
    batch_size: int,
    rounds: int,
    const_fn: Callable[[int], int],
) -> list[Instruction]:
    """Fully unrolled scalar main loop. Returns flat slots for packing by caller."""
    slots: list[Instruction] = []
    for round in range(rounds):
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

            # Compute next tree index and wrap
            slots.extend(build_parity_index(s, acc, cur_node))
            slots.append(("debug", ("compare", cur_node, (round, batch, "next_idx"))))
            slots.append(("alu", ("<", s.tmp1, cur_node, s.n_nodes)))
            slots.append(("flow", ("select", cur_node, s.tmp1, cur_node, s.zero_const)))
            slots.append(("debug", ("compare", cur_node, (round, batch, "wrapped_idx"))))
    return slots
