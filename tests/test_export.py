"""Tests for demo/export.py (no model needed)."""
import json

from demo.export import export_json, result_to_dict
from demo.solver import SolveResult

SOLVED = (
    "534678912672195348198342567859761423"
    "426853791713924856961537284287419635345286179"
)


def _result():
    traj = ["0" * 81, SOLVED]
    return SolveResult(final_board=SOLVED, trajectory=traj, num_steps=2)


def test_dict_shape():
    d = result_to_dict("0" * 81, _result())
    assert d["num_steps"] == 2
    assert d["solution"] == SOLVED
    assert len(d["trajectory"]) == 2
    assert set(d["stats"]) == {"stabilized_at", "first_valid_at", "changes_per_step"}


def test_json_roundtrip(tmp_path=None):
    payload = export_json("0" * 81, _result())
    parsed = json.loads(payload)
    assert parsed["puzzle"] == "0" * 81


def test_writes_file():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traj.json")
        export_json("0" * 81, _result(), path)
        assert os.path.exists(path)
        assert json.load(open(path))["num_steps"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
