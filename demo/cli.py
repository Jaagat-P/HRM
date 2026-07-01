"""Command-line HRM Sudoku solver.

Examples:
    python -m demo.cli --example hard
    python -m demo.cli --puzzle 000000010400000000...   # 81 chars
    python -m demo.cli --file my_puzzle.txt --trace

By default it downloads and runs the pretrained Sudoku-Extreme checkpoint and prints the
input board, the solved board, and a validity report.
"""
import argparse
import sys

from demo.checkpoint import load_hrm
from demo.examples import DEFAULT_EXAMPLE, EXAMPLES, get_example
from demo.solver import solve_puzzle
from demo.sudoku import format_board, parse_puzzle_string
from demo.validate import check_solution

DEFAULT_CHECKPOINT = "sapientinc/HRM-checkpoint-sudoku-extreme"


def _read_puzzle(args) -> str:
    if args.puzzle:
        return args.puzzle
    if args.file:
        with open(args.file) as f:
            return f.read()
    return get_example(args.example)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Solve a Sudoku with the Hierarchical Reasoning Model.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--example", choices=sorted(EXAMPLES), default=DEFAULT_EXAMPLE,
                     help="use a built-in example puzzle")
    src.add_argument("--puzzle", help="81-char puzzle string (0 or . for blanks)")
    src.add_argument("--file", help="path to a file containing an 81-char puzzle")

    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="HuggingFace repo id or local dir of the checkpoint")
    p.add_argument("--device", default=None, help="force a device (cpu/mps/cuda)")
    p.add_argument("--trace", action="store_true", help="print the board after every reasoning step")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    puzzle = _read_puzzle(args)

    print("Loading HRM checkpoint (first run downloads ~109 MB)...")
    loaded = load_hrm(args.checkpoint, device=args.device)
    print(f"Loaded on {loaded.device}.\n")

    normalized = "".join(str(d) for d in parse_puzzle_string(puzzle))
    print("Puzzle:")
    print(format_board(normalized))
    print()

    result = solve_puzzle(loaded, puzzle)

    if args.trace:
        for i, board in enumerate(result.trajectory, 1):
            print(f"--- step {i}/{result.num_steps} ---")
            print(format_board(board))
            print()

    print("Solution:")
    print(format_board(result.final_board))
    print(f"\nReasoning steps: {result.num_steps}")

    report = check_solution(puzzle, result.final_board)
    print(f"Complete & valid: {report.complete_valid}")
    print(f"Respects givens:  {report.respects_givens}")
    print("CORRECT" if report.correct else "INCORRECT")
    return 0 if report.correct else 1


if __name__ == "__main__":
    sys.exit(main())
