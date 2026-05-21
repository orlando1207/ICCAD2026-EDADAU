# FloorSet Usage Boundary

This note separates the official FloorSet files that can be used directly from the stages that must be implemented by us for contest or research work.

## Directly Usable Official Files

### Contest Driver And Scoring

Use these as-is:

- `FloorSet/iccad2026contest/iccad2026_evaluate.py`
  - Loads validation/training data through helper functions.
  - Imports an optimizer file dynamically.
  - Calls the optimizer `solve()` method.
  - Computes hard constraint feasibility, soft constraint penalties, HPWL, bounding-box area, runtime adjustment, and total score.
  - Provides CLI modes such as `--evaluate`, `--validate`, `--score`, `--baseline`, `--visualize`, and `--info`.
- `FloorSet/iccad2026contest/README.md`
  - Authoritative getting-started and rule summary for the contest sample.
- `FloorSet/iccad2026contest/FloorplanningContest_ICCAD_2026_v9.pdf`
  - Official problem statement.

Typical use:

```bash
cd FloorSet/iccad2026contest
python iccad2026_evaluate.py --evaluate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
python iccad2026_evaluate.py --validate my_optimizer.py
```

### Template Optimizer

Use as a starting point, not as final work:

- `FloorSet/iccad2026contest/optimizer_template.py`
  - Contains the required optimizer class pattern.
  - Provides a B*-tree simulated annealing baseline.
  - Shows how to preserve fixed/preplaced dimensions using `target_positions`.
  - Should usually be copied to a new file, for example `my_optimizer.py`, and then edited.

Typical use:

```bash
cd FloorSet/iccad2026contest
cp optimizer_template.py my_optimizer.py
```

### Data Loaders

Use these directly unless there is a clear reason to customize data access:

- `FloorSet/iccad2026contest/iccad2026_evaluate.py`
  - `get_training_dataloader(...)`
  - `get_validation_dataloader(...)`
- `FloorSet/liteLoader.py`
  - Training loader for FloorSet-Lite.
- `FloorSet/litetestLoader.py` and `FloorSet/lite_dataset_test.py`
  - Validation/test-style loader for FloorSet-Lite.
- `FloorSet/primeLoader.py`, `FloorSet/primetestLoader.py`, `FloorSet/prime_dataset.py`, `FloorSet/prime_dataset_test.py`
  - Legacy/general FloorSet-Prime loaders.

For contest work, prefer the helper functions in `iccad2026_evaluate.py` because they match the contest pipeline.

### Local Validation Data

Use directly for debugging and local evaluation:

- `FloorSet/LiteTensorDataTest/`
  - 100 validation samples.
  - `config_21` through `config_120`.
  - Each config contains one `litedata_1.pth` and one `litelabel_1.pth`.

Do not edit these `.pth` files. Treat them as immutable official inputs/labels.

### Cost, Constraint, And Visualization Utilities

Can be imported for analysis, debugging, or custom experiments:

- `FloorSet/cost.py`
  - Legacy/general wirelength and cost helpers.
- `FloorSet/utils.py`
  - Constraint-checking helpers.
- `FloorSet/visualize.py`
  - Layout visualization.
- `FloorSet/iccad2026contest/training_example.py`
  - Demonstrates differentiable training loss usage.

For actual contest scoring, rely on `iccad2026_evaluate.py` rather than reimplementing the scoring formula.

### Documentation And Reference Images

Use as references only:

- `FloorSet/README.md`
- `FloorSet/intel_testsuite.md`
- `FloorSet/intel_testsuite_lite.md`
- `FloorSet/images/`
- `FloorSet/inteltest_layouts/`
- `FloorSet/notebooks/`

These explain formats and show examples, but they are not the algorithm we need to submit.

## Stages We Need To Implement

### 1. Optimizer Strategy

Required.

We must implement the placement algorithm inside a `FloorplanOptimizer` subclass. The evaluator supplies one test case at a time and expects:

```python
[(x, y, width, height), ...]
```

with exactly `block_count` tuples.

Minimum viable strategy:

- Produce non-overlapping rectangles.
- Preserve every block area within 1 percent.
- Honor `target_positions` exactly for fixed-shape and preplaced blocks.
- Return quickly enough for all 100 validation cases.

Better strategy:

- Reduce block-to-block HPWL.
- Reduce pin-to-block HPWL.
- Reduce bounding-box area.
- Improve boundary, grouping, and MIB soft constraints.

### 2. Hard Constraint Handling

Required.

The optimizer must explicitly avoid infeasible output:

- No overlapping rectangles.
- `width * height` must match `area_targets[i]` within 1 percent for soft blocks.
- Fixed-shape blocks must use target `width,height`.
- Preplaced blocks must use target `x,y,width,height`.

Failing any hard constraint gives contest cost `10.0`, so this stage should come before fine-tuning HPWL.

### 3. Soft Constraint Handling

Important for score, but after hard feasibility.

The contest penalizes:

- Boundary constraints: required block touches bounding-box edge/corner.
- Grouping constraints: blocks in a group should abut and form one connected component.
- MIB constraints: blocks in the same MIB group should have identical dimensions.

The template baseline does not fully handle these. We need custom logic if we want competitive scores.

### 4. Objective Optimization

Required for quality after feasibility.

The scoring quality terms are based on:

- Block-to-block HPWL.
- Pin-to-block HPWL.
- Bounding-box area.
- Runtime factor.

Possible implementation families:

- Improve the provided B*-tree simulated annealing.
- Implement sequence pair, slicing tree, O-tree, corner block list, or another packing representation.
- Build constructive heuristics first, then local search.
- Train a model and use search/repair around predictions.

### 5. Training Pipeline

Optional unless using ML.

Official code provides data loading and differentiable loss helpers, but not a model. If using ML, we must implement:

- Model architecture.
- Feature encoding for areas, nets, pins, and constraints.
- Training loop, validation loop, checkpointing, and inference.
- Repair/postprocessing to satisfy hard constraints.

`training_example.py` is only a demonstration. It is not a complete model solution.

### 6. Experiment And Submission Workflow

Required for serious iteration.

We should maintain our own files for:

- Optimizer variants, for example `my_optimizer.py`, `optimizer_v2.py`, etc.
- Scripts to run selected validation IDs.
- Logs or CSV summaries of scores.
- Optional visualization outputs.

Use official validation before considering any optimizer usable:

```bash
cd FloorSet/iccad2026contest
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

## What Not To Implement First

Avoid spending early time on:

- Rewriting the official evaluator.
- Rewriting dataloaders.
- Modifying official `.pth` data.
- Optimizing training before a hard-feasible non-ML baseline exists.
- Relying only on differentiable loss without final evaluator checks.

The first useful milestone is a robust hard-feasible optimizer. The second is soft-constraint handling. The third is objective-quality improvement.

## Local Environment Notes

Current environment issue:

- `shapely` is missing, so official evaluator/loader imports fail until dependencies are installed.

Install dependencies:

```bash
pip install -r FloorSet/iccad2026contest/requirements.txt
```

Matplotlib may need a writable cache:

```bash
MPLCONFIGDIR=/tmp/matplotlib python FloorSet/iccad2026contest/iccad2026_evaluate.py --info
```
