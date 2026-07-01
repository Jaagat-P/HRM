"""Run all offline tests (no model download / GPU needed).

Run: python -m tests.run_all

Discovers test_* functions in the offline test modules and runs them, printing a summary.
The forward-pass smoke test (tests/test_forward_cpu.py) is included since it runs on CPU.
"""
import importlib
import traceback

OFFLINE_MODULES = [
    "tests.test_sudoku_encoding",
    "tests.test_validate",
    "tests.test_introspect",
    "tests.test_export",
    "tests.test_forward_cpu",
]


def main() -> int:
    passed = failed = 0
    for mod_name in OFFLINE_MODULES:
        module = importlib.import_module(mod_name)
        fns = {n: f for n, f in vars(module).items() if n.startswith("test_") and callable(f)}
        # Modules with a single test_forward use that name; otherwise run all test_*.
        for name, fn in sorted(fns.items()):
            try:
                fn()
                passed += 1
                print(f"PASS {mod_name}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
