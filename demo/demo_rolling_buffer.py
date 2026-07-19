# Compare rolling implementations for our use case
# Time save will be 100% negligible, but worth exploring
# Since we will apply arbitrary operations to the array later, keeping a
# contiguous array is likely better than messing with views.

# Findings:
# - In-place and no/reassign perform by far the best.
# - On larger buffer sizes, in-place suddenly improves wherever buffer size
# equals block size, and a slight improvement when exactly half. The former
# could be because shifts are ignored internally, but no clue for the latter.
# Reassigned concat is a bit more stable (on powers of 2 data, at least).
# - When without the 0/half-sized block advantage, reassigned concat overtakes
# in-place for buffer sizes >16318, even if the written block is tiny. At
# 65536, in-place can be as much as 50% slower (i.e. concat is ~30% faster).
# - Reassign probably creates a new ref, so is only very slightly slower than
# no assignment at all (concat still creates a new array, rather than no-op).
# - Broadcasted concat still improves over np.roll despite the redundant copy,
# but np.roll can sometimes outperform out of nowhere.
# - np.roll consistently performs poorly, ranging 2-10x that of the best.
# So it probably should never be used unless a roll is exactly what is needed
# (and no other specialized lib function exists).

import timeit

import numpy as np

if __name__ == "__main__":
    # We know our use case only uses powers of 2
    initial_buffer_size = 2048
    initial_block_size = 1024
    n_buffer_sizes = 6  # Test up to 65536 sizes

    statement_map = {
        # Uses np.roll to rotate back, but unclear whether it in-place swaps or uses an auxiliary array 
        "np.roll": "np.roll(buffer, -block_size)\nbuffer[-block_size:] = block",
        # Manual indexing to skip shifting head items
        # timeit cannot write to the outer var, so using broadcast
        "concat_broadcast": "buffer[:] = np.concat([buffer[:-block_size], block])",
        # Ditto, but remove the forced copy
        "concat_no_assign": "np.concat([buffer[:-block_size], block])",
        "concat_reassign": "a = np.concat([buffer[:-block_size], block])",
        # In-place shift to avoid new array overhead, but back to 2 lines
        "in_place": "buffer[:-block_size] = buffer[block_size:]\nbuffer[-block_size:] = block",
    }
    repeats = 10000
    rng = np.random.default_rng()

    # Generate n(n-1) tests where buffer >= stride
    for i in range(n_buffer_sizes):
        for j in range(i + 2):
            buffer_size = 2**i * initial_buffer_size
            block_size = 2**j * initial_block_size

            block = rng.standard_normal(block_size)

            # Test against each rolling impl
            for stmt_name, stmt in statement_map.items():
                # Let it mutate. Expect identical blocks by the end.
                buffer = rng.standard_normal(buffer_size)

                timing = timeit.timeit(
                    stmt,
                    globals=globals(),
                    number=repeats,
                )
                print(f"Buffer: {buffer_size}, Block: {block_size}, Stmt: {stmt_name}, Timing (ms): {1000 / repeats * timing:.5f}")
            print()
