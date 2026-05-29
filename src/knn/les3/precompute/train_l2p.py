#!/usr/bin/env python3

import argparse
import json
import sys

from l2p_core import train_l2p


def main():
    parser = argparse.ArgumentParser(description="Train LES3 L2P groups from a set file and PTR embedding.")
    parser.add_argument("--input", required=True, help="Input set file")
    parser.add_argument("--ptr", required=True, help="Input PTR representation file")
    parser.add_argument("--output-dir", required=True, help="Directory for level and final LES3 group files")
    parser.add_argument("--label", required=True, help="Label for the run, printed in json")
    args = parser.parse_args()

    try:
        result = train_l2p(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    json.dump(result, sys.stdout, indent=4)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
