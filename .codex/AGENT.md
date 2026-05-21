# Project Agent Notes

## Repository Shape

- Root project: `/home/orlando/ICCAD2026-EDADAU`.
- Official sample bundle lives in `FloorSet/` with a capital `F`.
- `FloorSet/` is currently untracked by the outer git repository and also contains its own nested `.git/` clone from `https://github.com/IntelLabs/FloorSet.git`.
- Treat `FloorSet/` as vendor/official sample code unless the user explicitly asks to modify it. Prefer creating contest solutions as separate files under `FloorSet/iccad2026contest/` copied from `optimizer_template.py`.

## FloorSet Contents

- Total local size: about `384M`.
- Largest local directories:
  - `FloorSet/LiteTensorDataTest/`: about `256M`; 100 validation configurations, `config_21` through `config_120`, each with `litedata_1.pth` and `litelabel_1.pth`.
  - `FloorSet/.git/`: about `100M`; nested upstream repo history.
  - `FloorSet/images/`: about `12M`; README figures and GIFs.
  - `FloorSet/inteltest_layouts/`: about `6.9M`; 100 PNG layout visualizations plus README/GIF.
  - `FloorSet/data/`: small local archive, currently `PrimeTensorDataTest.tar.gz`.
- File-type counts observed locally: 200 `.pth`, 113 `.png`, 28 `.py`, 6 `.md`, 5 `.ipynb`, 3 `.gif`, 1 `.pdf`, plus git internals.

## Official Contest Entry Point

Work from:

```bash
cd FloorSet/iccad2026contest
```

Useful commands:

```bash
cp optimizer_template.py my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --info
```

Main files:

- `iccad2026_evaluate.py`: authoritative contest evaluator, dataloader helpers, scoring, validation, visualization hooks, differentiable training losses.
- `optimizer_template.py`: B*-tree simulated annealing baseline. Copy this and replace `solve()`.
- `training_example.py`: example use of the differentiable training loss.
- `FloorplanningContest_ICCAD_2026_v9.pdf`: contest problem statement.
- `README.md`: contest-specific rules and changelog.

See `codex/FLOORSET_USAGE_BOUNDARY.md` for the practical boundary between official files that can be used directly and stages that must be implemented locally.

## Use Directly Vs Implement Locally

Use directly:

- `FloorSet/iccad2026contest/iccad2026_evaluate.py` for evaluation, validation, scoring, dataloader helpers, baseline extraction, and CLI commands.
- `FloorSet/iccad2026contest/optimizer_template.py` as a starting template only; copy it before editing.
- `FloorSet/LiteTensorDataTest/` as immutable local validation data.
- Official loaders in `liteLoader.py`, `litetestLoader.py`, and `iccad2026_evaluate.py` unless custom data access is necessary.
- `cost.py`, `utils.py`, and `visualize.py` for debugging/analysis, while keeping final scoring delegated to `iccad2026_evaluate.py`.

Implement locally:

- The optimizer/search/ML algorithm inside a `FloorplanOptimizer` subclass.
- Hard-feasibility logic: no overlaps, area within 1 percent, exact fixed-shape dimensions, exact preplaced position/dimensions.
- Soft-constraint handling for boundary, grouping, and MIB.
- Objective optimization for block-to-block HPWL, pin-to-block HPWL, bounding-box area, and runtime.
- Any ML model architecture, feature encoding, training loop, checkpointing, inference, and repair/postprocessing.

Implementation priority:

1. Build a robust hard-feasible non-overlapping baseline.
2. Add fixed/preplaced handling using `target_positions`.
3. Add soft-constraint logic for boundary, grouping, and MIB.
4. Improve HPWL and area through search, packing, or ML-assisted placement.
5. Run `--validate`, selected `--test-id`, then full `--evaluate`.

## Current Baseline Optimizer

`FloorSet/baseline_optimizer.py` now implements a constraint-aware hierarchical
B*-tree simulated annealing floorplanner:

- Soft block shape is represented by target area plus aspect ratio, initialized
  with `AR = 1.0`.
- Soft MIB groups share one aspect-ratio state and use the average group area.
- Preplaced blocks are excluded from B*-trees and inserted into the contour as
  immutable obstacles.
- Fixed-shape blocks keep exact target dimensions for v9 evaluator
  compatibility.
- Cluster/group constraints are represented as sub-B*-tree macros during the
  high-temperature SA stage, then expanded into individual blocks during the
  low-temperature stage.
- High-stage cost emphasizes HPWL, bounding-box area, and boundary distance.
- Low-stage cost adds grouping dead-space and official-style soft-violation
  proxies.

The detailed workflow is documented in:

```bash
.codex/BSTAR_SA_IMPLEMENTED_WORKFLOW.md
```

## Optimizer Contract

Implement a class that subclasses `FloorplanOptimizer`, usually by editing a copy of `optimizer_template.py`.

`solve()` receives:

```python
solve(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
)
```

Return exactly `block_count` tuples:

```python
[(x, y, width, height), ...]
```

Important constraints from v9:

- Hard infeasible, cost `10.0`: block overlaps, area outside 1 percent tolerance, fixed-shape dimension mismatch, preplaced dimension or location mismatch.
- Soft penalties: grouping, MIB, boundary.
- Relaxed: aspect ratio, fixed outline, integer coordinates.
- Use `target_positions` when present. Fixed blocks provide target `w,h`; preplaced blocks provide target `x,y,w,h`.

## Dataset Format

Top-level README describes the general FloorSet format. For the contest Lite validation set in this local copy:

- Raw files are under `FloorSet/LiteTensorDataTest/config_N/`.
- `N` is the block count from 21 to 120.
- Each config has one `litedata_1.pth` and one `litelabel_1.pth`.
- `lite_dataset_test.py` converts raw data as:
  - `area_target = raw[0][:, 0]`
  - `placement_constraints = raw[0][:, 1:]`
  - `b2b_connectivity = raw[1]`
  - `p2b_connectivity = raw[2]`
  - `pins_pos = raw[3]`
  - labels are `(fp_sol, metrics)`, where `metrics` has 8 values.
- Constraint columns are `[fixed, preplaced, MIB, cluster/grouping, boundary]`.

The official dataloaders can auto-download missing data from Hugging Face, which may require network approval in this sandbox.

## Dependencies And Local Gotchas

Install contest dependencies with:

```bash
pip install -r FloorSet/iccad2026contest/requirements.txt
```

Known current environment issue:

- `shapely` is not installed, so importing official dataloaders/evaluator currently fails at runtime.
- Matplotlib may warn that `/home/orlando/.config/matplotlib` is not writable; set `MPLCONFIGDIR=/tmp/matplotlib` for commands that import plotting code.

Example:

```bash
MPLCONFIGDIR=/tmp/matplotlib python FloorSet/iccad2026contest/iccad2026_evaluate.py --info
```

## Development Guidance

- Use `rg`/`rg --files` for navigation.
- Avoid shuffling dataloaders unless needed; the official README notes shuffling can slow file caching.
- Do not run large training downloads casually. Training data is much larger than the local validation set.
- Do not delete or rewrite the nested official `FloorSet/.git/` unless the user asks.
- Before changing vendor files, check `git status --short` in both the outer repo and `FloorSet/`.
