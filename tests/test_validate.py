"""Tests for demo/validate.py."""
from demo.validate import check_solution, is_complete_valid, respects_givens

SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)
PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)


def test_valid_complete_solution():
    assert is_complete_valid(SOLUTION)


def test_incomplete_is_invalid():
    assert not is_complete_valid(PUZZLE)  # has blanks


def test_row_violation_detected():
    bad = "1" + SOLUTION[1:]  # duplicate 1 in first row/col
    assert not is_complete_valid(bad)


def test_respects_givens_true():
    assert respects_givens(PUZZLE, SOLUTION)


def test_respects_givens_false_when_given_changed():
    # Change the first given (5 -> 6) in the solution.
    tampered = "6" + SOLUTION[1:]
    assert not respects_givens(PUZZLE, tampered)


def test_check_solution_correct():
    report = check_solution(PUZZLE, SOLUTION)
    assert report.correct


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
