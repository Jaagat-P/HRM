"""Export an HRM solve result to JSON for external tools (e.g. the web visualizer)."""
import json
from typing import Optional

from demo.introspect import analyze
from demo.solver import SolveResult


def result_to_dict(puzzle: str, result: SolveResult) -> dict:
    stats = analyze(result)
    return {
        "puzzle": puzzle,
        "solution": result.final_board,
        "num_steps": result.num_steps,
        "trajectory": result.trajectory,
        "stats": {
            "stabilized_at": stats.stabilized_at,
            "first_valid_at": stats.first_valid_at,
            "changes_per_step": stats.changes_per_step,
        },
    }


def export_json(puzzle: str, result: SolveResult, path: Optional[str] = None) -> str:
    """Serialize a solve result to JSON; write to `path` if given. Returns the JSON string."""
    payload = json.dumps(result_to_dict(puzzle, result), indent=2)
    if path:
        with open(path, "w") as f:
            f.write(payload)
    return payload
