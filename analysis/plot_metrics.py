from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.experiment_utils import plot_training_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.dirname(args.csv_path)
    os.makedirs(out_dir, exist_ok=True)
    plot_training_metrics(args.csv_path, out_dir)


if __name__ == "__main__":
    main()
