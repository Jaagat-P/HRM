# Does HRM spend more reasoning on harder puzzles?

A small observation from running the pretrained Sudoku-Extreme checkpoint through the demo's
introspection tooling (`demo/introspect.py`, `demo/benchmark.py`).

HRM refines its answer across up to 16 reasoning steps. We measure **stabilization** — the step
after which the predicted board stops changing — as a proxy for how much reasoning each puzzle
required. Fewer clues (givens) means a harder puzzle.

| Puzzle | Givens | Stabilized at step | First valid at step | Correct |
|--------|:------:|:------------------:|:-------------------:|:-------:|
| easy   |   30   |         2          |          2          |   ✅    |
| medium |   24   |         4          |          4          |   ✅    |
| hard   |   17   |         5          |          5          |   ✅    |
| blank  |    0   |        16          |        never        |   ❌    |

## Takeaways

- **Monotonic trend:** as the number of givens drops (30 → 24 → 17), the model uses more
  reasoning steps to stabilize (2 → 4 → 5). The model appears to allocate more computation to
  harder inputs, consistent with the Adaptive Computation Time design.
- **First-valid == stabilized** in every solved case: HRM does not reach a valid grid early and
  then keep editing; it stabilizes exactly when it becomes correct.
- **The blank board never converges** and is invalid — with zero constraints there is no unique
  target, so the model keeps changing cells until the step cap.

## Reproduce

```bash
python -m demo.benchmark
```

## Caveats

This is a handful of puzzles, not a controlled study. A proper analysis would sweep many
puzzles across a difficulty rating (the Sudoku-Extreme dataset ships a per-puzzle rating) and
correlate rating with stabilization step. That's a natural next extension.
