"""Tests for demo/introspect.py using synthetic trajectories (no model needed)."""
from demo.introspect import analyze
from demo.solver import SolveResult

SOLVED = (
    "534678912672195348198342567859761423"
    "426853791713924856961537284287419635345286179"
)


def _result(traj):
    return SolveResult(final_board=traj[-1], trajectory=traj, num_steps=len(traj))


def test_stabilizes_early():
    # Changes on step 2, then never again.
    traj = ["0" * 81, SOLVED, SOLVED, SOLVED]
    stats = analyze(_result(traj))
    assert stats.stabilized_at == 2
    assert stats.first_valid_at == 2
    assert stats.changes_per_step == [0, 81, 0, 0]


def test_late_stabilization():
    traj = ["0" * 81, "1" * 81, SOLVED, SOLVED]
    stats = analyze(_result(traj))
    assert stats.stabilized_at == 3


def test_never_valid():
    traj = ["0" * 81, "1" * 81]
    stats = analyze(_result(traj))
    assert stats.first_valid_at is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
