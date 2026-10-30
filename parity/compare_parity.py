#!/usr/bin/env python3
"""Parity diff: compare two directories of plp solver JSONs.

Every deterministic JSON field must be identical (including stability_score
floats); time_s is excluded — machine-dependent.

usage: compare_parity.py DIR_A DIR_B [NAME ...]
    NAME = json stem (e.g. BR1); default: all *.json in DIR_A.
Exit 0 iff every instance MATCHES.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    da, db = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    if not names:
        names = sorted(f[:-5] for f in os.listdir(da) if f.endswith(".json"))
    ok = True
    for n in names:
        a = json.load(open(os.path.join(da, n + ".json")))
        b = json.load(open(os.path.join(db, n + ".json")))
        a.pop("time_s")
        b.pop("time_s")
        if a == b:
            print(n, "MATCH")
            continue
        ok = False
        print(n, "MISMATCH")
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print(f"  field: {k}")
                print("    A:", str(a.get(k))[:200])
                print("    B:", str(b.get(k))[:200])
    print("RESULT:", "ALL MATCH" if ok else "HAS MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
