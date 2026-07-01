# HRM Sudoku Demo (runs on CPU / Apple Silicon)

Run the pretrained Hierarchical Reasoning Model to solve Sudoku puzzles on a laptop — no
NVIDIA GPU or FlashAttention required. The model falls back to PyTorch's native attention
when FlashAttention is unavailable.

## Quick start

```bash
pip install torch huggingface_hub pyyaml einops pydantic
python -m demo.cli --example hard
```

The first run downloads the ~109 MB pretrained checkpoint
(`sapientinc/HRM-checkpoint-sudoku-extreme`) from Hugging Face.

## Usage

```bash
python -m demo.cli --example easy           # built-in puzzle (easy | medium | hard | blank)
python -m demo.cli --puzzle 0000000104...   # 81-char string, 0 or . for blanks
python -m demo.cli --file puzzle.txt        # read a puzzle from a file
python -m demo.cli --example hard --animate # watch the solution emerge step by step
python -m demo.cli --example hard --trace   # print every reasoning step
python -m demo.cli --example hard --export-json traj.json  # dump trajectory for the visualizer
```

### More tools

```bash
python -m demo.benchmark          # solve all examples, tabulate reasoning steps vs difficulty
python -m demo.batch puzzles.txt  # solve one puzzle per line, report accuracy
python -m demo.serve              # serve the web visualizer at http://localhost:8000/visualizer.html
python -m tests.run_all           # run the offline test suite
```

See [`FINDINGS.md`](./FINDINGS.md) for an observation on reasoning steps vs. puzzle difficulty.

## Modules

| File | Purpose |
|------|---------|
| `sudoku.py` | encode/decode between puzzle strings and model tokens |
| `checkpoint.py` | load a pretrained checkpoint on any device (cuda > mps > cpu) |
| `solver.py` | run the ACT loop, capturing the step-by-step trajectory |
| `validate.py` | verify a solution is a valid, givens-preserving Sudoku |
| `introspect.py` | summarize how the trajectory evolved (steps to stabilize, etc.) |
| `animate.py` | replay the trajectory in the terminal |
| `export.py` | serialize a solve result to JSON |
| `benchmark.py` | solve all examples, report steps vs. difficulty |
| `batch.py` | solve many puzzles from a file |
| `serve.py` | serve the web visualizer locally |
| `cli.py` | command-line entry point |
| `examples.py` | built-in example puzzles |
| `web/visualizer.html` | browser trajectory visualizer |
