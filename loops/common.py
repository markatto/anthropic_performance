"""Shared building blocks for main loop implementations."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from problem import HASH_STAGES

if TYPE_CHECKING:
    from kernel import ScratchLayout

Slot = tuple
Instruction = tuple[str, Slot]  # ("engine", (op, ...))

# Re-export for type annotations in other modules
__all__ = ["Instruction", "Slot", "build_hash", "build_parity_index"]


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


def build_parity_index(
    s: "ScratchLayout",  # noqa: F821 - forward ref to avoid circular import
    acc: int,
    cur_node: int,
) -> list[Instruction]:
    """Compute next tree index from hash parity: idx = 2*idx + (1 if hash%2==0 else 2)."""
    return [
        ("alu", ("%", s.tmp1, acc, s.two_const)),
        ("alu", ("+", s.tmp3, s.tmp1, s.one_const)),
        ("alu", ("*", cur_node, cur_node, s.two_const)),
        ("alu", ("+", cur_node, cur_node, s.tmp3)),
    ]
