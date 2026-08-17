"""Command-line interface for job_classifier."""

import argparse
import csv
from pathlib import Path

DEFAULT_RAW_POSITIONS_PATH = Path("data/raw_positions.csv")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a `;`-delimited, UTF-8-BOM CSV file into a list of dict rows."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def cmd_match(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.input)
    print(len(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job_classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    match_parser = subparsers.add_parser("match", help="Match raw job titles against the classifier")
    match_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_POSITIONS_PATH,
        help="Path to raw_positions.csv (default: data/raw_positions.csv)",
    )
    match_parser.set_defaults(func=cmd_match)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
