"""Solve all built-in examples and report reasoning effort vs. correctness.

Run: python -m demo.benchmark

Prints a table of steps-to-stabilize and validity for each example puzzle - a quick look at
whether the model spends more reasoning steps on harder inputs.
"""
from demo.checkpoint import load_hrm
from demo.examples import EXAMPLES
from demo.introspect import analyze
from demo.solver import solve_puzzle
from demo.validate import check_solution

DEFAULT_CHECKPOINT = "sapientinc/HRM-checkpoint-sudoku-extreme"


def run(checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
    loaded = load_hrm(checkpoint, device=device)
    print(f"Loaded on {loaded.device}\n")

    header = f"{'puzzle':<8} {'givens':>6} {'stabilized':>10} {'first_valid':>11} {'correct':>7}"
    print(header)
    print("-" * len(header))

    for name, puzzle in EXAMPLES.items():
        result = solve_puzzle(loaded, puzzle)
        stats = analyze(result)
        report = check_solution(puzzle, result.final_board)
        givens = sum(1 for c in puzzle if c != "0")
        print(f"{name:<8} {givens:>6} {str(stats.stabilized_at):>10} "
              f"{str(stats.first_valid_at):>11} {str(report.correct):>7}")


if __name__ == "__main__":
    run()
