"""Tests for demo/sudoku.py encode/decode round-tripping."""
from demo.sudoku import decode_board, encode_puzzle, format_board, parse_puzzle_string

# A real Sudoku puzzle (0 = blank) and its known solution.
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


def test_parse_length_and_blanks():
    digits = parse_puzzle_string(PUZZLE)
    assert len(digits) == 81
    assert digits[2] == 0  # third cell is blank
    assert digits[0] == 5


def test_dot_and_zero_are_equivalent_blanks():
    dotted = PUZZLE.replace("0", ".")
    assert parse_puzzle_string(dotted) == parse_puzzle_string(PUZZLE)


def test_encode_decode_roundtrip():
    tokens = encode_puzzle(PUZZLE)
    assert tokens.shape == (1, 81)
    # Tokens are digit + 1, so range is 1..10.
    assert int(tokens.min()) >= 1 and int(tokens.max()) <= 10
    assert decode_board(tokens) == PUZZLE


def test_whitespace_is_ignored():
    spaced = "\n".join(PUZZLE[i:i + 9] for i in range(0, 81, 9))
    assert decode_board(encode_puzzle(spaced)) == PUZZLE


def test_format_board_shape():
    grid = format_board(PUZZLE)
    lines = grid.splitlines()
    # 9 rows + 2 box separator lines
    assert len(lines) == 11


def test_invalid_length_raises():
    try:
        parse_puzzle_string("123")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nExample board:\n")
    print(format_board(PUZZLE))
