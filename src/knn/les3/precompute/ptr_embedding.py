#!/usr/bin/env python3

import argparse
import json
import sys

from l2p_core import ptr_embedding


def main():
    parser = argparse.ArgumentParser(description="Build LES3 PTR representations from a set file.")
    parser.add_argument("--input", required=True, help="Input set file")
    parser.add_argument("--output", required=True, help="Output PTR representation file")
    args = parser.parse_args()

    try:
        result = ptr_embedding(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    json.dump(result, sys.stdout, indent=4)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
