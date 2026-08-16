# Repository Guidelines

## Project Structure & Module Organization

FloorSet is a script-oriented Python repository for VLSI floorplanning datasets. Root modules are split into parallel Prime and Lite paths: `prime_dataset.py`/`lite_dataset.py` define datasets, while `primeLoader.py`, `liteLoader.py`, and their `*testLoader.py` counterparts provide loaders. Shared scoring and geometry logic lives in `cost.py`, `utils.py`, and `validate.py`; rendering is in `visualize.py`. Dataset checks are `prime_dataset_test.py` and `lite_dataset_test.py`. Large tensor data belongs under `PrimeTensorData/`, `LiteTensorDataTest/`, or `floorset_lite/`; generated figures belong in `images/` or `inteltest_layouts/`. The `iccad2026contest/` directory is a separate contest workflow with its own README and guidance.

## Build, Test, and Development Commands

There is no build step; run modules from this directory with Python 3.

```bash
python -m pip install -r requirements.txt
python -m compileall -q *.py
python prime_dataset_test.py
python lite_dataset_test.py
python validate.py
cd iccad2026contest && python iccad2026_evaluate.py --evaluate my_optimizer.py
```

The compile command is a quick syntax check. Dataset tests and `validate.py` exercise loading, collation, and scoring, but may prompt for multi-gigabyte downloads when data is absent. The final command runs the official contest evaluator; consult `iccad2026contest/README.md` before changing an optimizer.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep Prime and Lite behavior aligned when changing shared concepts, while respecting their different solution formats. Add short docstrings for public helpers and comments only where tensor shapes or constraint semantics are not obvious. No formatter or linter is configured, so keep imports grouped and avoid unrelated reformatting.

## Testing Guidelines

Tests are executable scripts rather than pytest suites. Name new checks `*_test.py`, make failures explicit with assertions, and cover variable-size batching and `-1` padding. For scoring changes, test wirelength plus fixed, preplaced, MIB, cluster, and boundary constraints. Use small local fixtures where possible to avoid network-dependent tests.

## Commit & Pull Request Guidelines

History uses brief, descriptive subjects; prefer an imperative summary such as `Fix Lite boundary validation`. Keep commits scoped and exclude downloaded tensors, archives, checkpoints, caches, and generated results. Pull requests should explain the affected Prime/Lite/contest path, list commands run, note dataset requirements, link relevant issues, and include before/after images for visualization or placement changes.
