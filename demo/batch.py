"""Solve many Sudoku puzzles from a file.

Run: python -m demo.batch puzzles.txt [--out solutions.txt]

The input file has one 81-char puzzle per line (blank lines and lines starting with '#'
are ignored). Prints per-puzzle correctness and an accuracy summary; optionally writes the
solved boards to an output file.
"""
import argparse
from typing import List

from demo.checkpoint import load_hrm
from demo.solver import solve_puzzle
from demo.validate import check_solution

DEFAULT_CHECKPOINT = "sapientinc/HRM-checkpoint-sudoku-extreme"


def read_puzzles(path: str) -> List[str]:
    puzzles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                puzzles.append(line)
    return puzzles


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-solve Sudoku puzzles with HRM.")
    parser.add_argument("input", help="file with one 81-char puzzle per line")
    parser.add_argument("--out", help="optional file to write solved boards to")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    puzzles = read_puzzles(args.input)
    print(f"Loaded {len(puzzles)} puzzles. Loading model...")
    loaded = load_hrm(args.checkpoint, device=args.device)

    solutions, correct = [], 0
    for i, puzzle in enumerate(puzzles, 1):
        result = solve_puzzle(loaded, puzzle)
        report = check_solution(puzzle, result.final_board)
        correct += report.correct
        solutions.append(result.final_board)
        print(f"[{i}/{len(puzzles)}] {'OK ' if report.correct else 'BAD'} steps={result.num_steps}")

    print(f"\nAccuracy: {correct}/{len(puzzles)} ({100 * correct / max(len(puzzles), 1):.1f}%)")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(solutions) + "\n")
        print(f"Wrote solutions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
