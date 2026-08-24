# Beta Evaluation Feasibility Issue

## Summary

The Beta submission completed all 100 hidden proxy cases without runtime
exceptions, but only 29 results were hard-feasible:

- Total score: **7.2710**
- Feasible cases: **29/100**
- Average cost: **7.4427**
- Average runtime: **1.193 s**
- Mean cost among feasible cases: **1.1819**
- Median cost among feasible cases: **1.1548**

The feasible-case quality confirms that the prediction and optimization
pipeline remains competitive. The dominant problem is that the legalizer does
not guarantee all official hard constraints on hidden-case constraint
combinations.

## Evidence

- Every Beta result has `error: null`; this is a geometry-feasibility failure,
  not an execution failure.
- Twelve of the 71 infeasible cases have zero soft-constraint violations.
  Boundary, grouping, and MIB soft penalties therefore do not explain hard
  infeasibility.
- Failures occur at small sizes (including 23 and 25 blocks), while some cases
  with 117, 119, and 120 blocks are feasible. Complexity increases risk but is
  not the root cause.
- The local result was 100/100 feasible with the same general pipeline,
  indicating that hidden cases exercise constraint combinations not adequately
  covered by the local validation set.

The released Beta JSON does not contain returned positions or individual hard
violation counts. It is therefore impossible to determine the exact split
among overlap, area, and immutable-geometry violations from the results alone.

## Official hard-feasibility conditions

A solution is feasible only when all of the following hold:

1. No pair of blocks overlaps by more than `1e-6` on both axes.
2. Every ordinary soft block has area within 1% of its target.
3. Every fixed block preserves its specified width and height within `1e-4`.
4. Every preplaced block preserves its position, width, and height within
   `1e-4`.

## Confirmed implementation risks

### 1. The candidate proxy checks only overlap

`floordiff/legalizer.py::proxy_cost()` assigns cost 10 for overlap, but does
not check soft-block area tolerance, fixed dimensions, or preplaced geometry.
A candidate can therefore win internal selection and still be rejected by the
official evaluator.

### 2. Graph-repair failure is ignored

`legal_round()` discards the `_ok` result from both `repair_graph()` calls. The
last wall-free `do_assign(False, False)` result is also ignored. Consequently,
the method can return a solution even after every overlap-free assignment
attempt failed.

This is particularly dangerous when separation graphs conflict with multiple
immutable preplaced anchors.

### 3. There is no final hard-feasibility gate

The optimizer does not run an official-equivalent hard validator immediately
before returning. It also has no guaranteed deterministic rescue when all
sampled candidates remain infeasible.

Parallel selection returns the minimum proxy candidate even if all candidates
have proxy cost 10.

### 4. MIB tying can violate area or immutable dimensions

The legalizer assigns every MIB member the representative member's width and
height. It does not verify afterward that:

- every member remains within its own target-area tolerance;
- multiple fixed/preplaced members have compatible dimensions; or
- immutable members were not changed by group tying.

This may not occur in the local data but can fail on more varied hidden
constraint combinations.

### 5. Area scaling has limited safety margin

Ordinary soft blocks use `area_scale = 0.991`, only 0.1 percentage points
inside the 1% tolerance. Reshaping is intended to preserve this scaled area,
but all final candidates should still be checked using the exact official
formula.

## Development priorities

### P0: Make feasibility observable

Implement one internal hard validator matching the official evaluator. For
each candidate, record:

- overlap count and maximum penetration;
- soft-block area violations and maximum relative area error;
- fixed-dimension violations;
- preplaced position/dimension violations;
- affected block and pair indices.

Use this validator both after each legalization stage and immediately before
returning the solution.

### P0: Select feasible candidates first

Candidate ordering must be lexicographic:

1. hard-feasible before hard-infeasible;
2. official quality/soft-constraint proxy among feasible candidates;
3. hard-violation severity only when no feasible candidate exists.

Never allow HPWL or area quality to outrank hard feasibility.

### P0: Add a guaranteed fallback

If all learned candidates fail, invoke a deterministic construction that:

- stamps all fixed/preplaced geometry exactly;
- treats preplaced blocks as immutable obstacles;
- gives ordinary soft blocks legal target-area dimensions;
- places remaining blocks without overlap;
- prioritizes feasibility over HPWL and bounding-box quality.

Any feasible fallback is preferable to the fixed infeasibility cost of 10.

### P1: Honor graph status

- Propagate every `repair_graph()` success flag.
- Treat failed `do_assign()` calls as failed candidates.
- Rebuild the separation graph or invoke fallback instead of returning a
  partially repaired layout.
- Revalidate after polish, snapping, cluster alignment, and cluster repair.

### P1: Harden immutable and MIB handling

- Re-stamp fixed/preplaced dimensions and preplaced positions at the end.
- Repair only movable blocks after re-stamping.
- Validate target-area compatibility before tying an MIB group.
- Handle multiple immutable MIB members explicitly instead of blindly choosing
  the first representative.
- Increase the ordinary-area safety margin if numerical testing warrants it.

### P1: Add adversarial tests

Create synthetic and transformed cases covering:

- multiple preplaced anchors on conflicting sides;
- fixed + MIB and preplaced + MIB combinations;
- groups containing multiple immutable members;
- boundary-constrained preplaced blocks;
- dense 100--120 block predictions;
- inconsistent or nearly inconsistent separation graphs;
- candidates far outside the local training distribution.

Every test should assert all official hard checks independently of the quality
score.

## Recommended workflow

1. Add diagnostic hard checks without changing placement behavior.
2. Run all local cases and adversarial cases, collecting violation categories.
3. Fix candidate selection and ignored graph failures.
4. Implement and verify the deterministic feasibility fallback.
5. Harden MIB/immutable interactions.
6. Only after achieving robust 100% feasibility, resume HPWL, area, soft-
   constraint, and runtime optimization.

The next milestone should be **hard-feasible output for every tested input**, not
a lower average score on the already-feasible subset.
