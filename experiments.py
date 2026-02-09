"""
Experimental code fragments for exploring the architecture.

Run tests with: pytest experiments.py -v
"""

from problem import (
    Machine,
    DebugInfo,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
)

from problem import HASH_STAGES, myhash


def run_fragment(instrs, mem=None, scratch_size=SCRATCH_SIZE):
    """
    Helper to run a code fragment and return the final machine state.
    
    Args:
        instrs: list of instruction bundles
        mem: initial memory image (defaults to 1024 zeros)
        scratch_size: size of scratch space
    
    Returns:
        Machine instance after execution
    """
    if mem is None:
        mem = [0] * 1024
    
    debug_info = DebugInfo(scratch_map={})
    machine = Machine(
        mem,
        instrs,
        debug_info,
        n_cores=N_CORES,
        scratch_size=scratch_size,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine



def hash(val):
    '''hacked up and recreated version of hash fn for testing'''
    hash_addr = 0
    tmp1, tmp2 = 1, 2  # scratch addresses
    

    const_base = 3

    instructions = [
        {"load": [("const", hash_addr, val)]},  # scratch[0] = val
    ]    
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        cb = const_base + 2*hi
        val1_addr = cb
        val3_addr = cb + 1

        instructions.append({"load": [("const", val1_addr, val1)]})
        instructions.append({"load": [("const", val3_addr, val3)]})
        instructions.append({"alu": [(op1, tmp1, hash_addr, cb)]})
        instructions.append({"alu": [(op3, tmp2, hash_addr, cb+1)]})
        instructions.append({"alu": [(op2, hash_addr, tmp1, tmp2)]})


    machine = run_fragment(instructions)
    return machine.cores[0].scratch[0]

def simple_hash(a: int) -> int:
    """
    hardcoded version of the hash function
    in normal infix python for my smooth brain
    """
    def r(x):
        return x % (2**32)

    a = r(r(a + 0x7ED55D16) + r(a << 12)) 
    a = r(r(a ^ 0xC761C23C) ^ r(a >> 19))
    a = r(r(a + 0x165667B1) + r(a << 5))
    a = r(r(a + 0xD3A2646C) ^ r(a << 9))
    a = r(r(a + 0xFD7046C5) + r(a << 3))
    a = r(r(a ^ 0xB55A4F09) ^ r(a >> 16))
    return a

def test_hash():
    
    assert hash(10) == 1712784324
    assert myhash(10) == 1712784324
    assert simple_hash(10) == 1712784324
