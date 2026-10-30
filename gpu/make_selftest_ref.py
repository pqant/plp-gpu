#!/usr/bin/env python
"""Generate the --selftest reference for the plp_gpu C++ port.

Protocol (must match plp_gpu.cpp selftest() EXACTLY):
  Block 1 (RNG): for each seed in SEEDS: 16 x random() as 16-hex-digit
    IEEE-754 bit pattern, then 16 x getrandbits(13) decimal.
  Block 2 (repr): for each value in VALS: repr(value) — CPython's float
    repr is the JSON formatting target (plp::py_repr must match it).

Run with the repo venv python (3.11):
  cd experiments/plp && ../../packages/paper-forge/.venv/bin/python gpu/make_selftest_ref.py > gpu/selftest_ref.txt
"""
import struct

SEEDS = (0, 1, 7, 123456789)

VALS = (
    0.0, 1.0, 0.5, 0.1, 2.0 / 3.0, 0.5785123966942148,
    0.6132075471698113, 0.6595744680851063, 1e-05, 1e-04,
    1e15, 1e16, 0.001, 123456789.5, 1.0 / 121.0, 1.0 / 106.0,
)


def main() -> None:
    import random
    for s in SEEDS:
        print(f"SEED {s}")
        r = random.Random(s)
        for _ in range(16):
            bits = struct.unpack("<Q", struct.pack("<d", r.random()))[0]
            print(format(bits, "016x"))
        for _ in range(16):
            print(r.getrandbits(13))
    for v in VALS:
        print(f"REPR {repr(v)}")


if __name__ == "__main__":
    main()
