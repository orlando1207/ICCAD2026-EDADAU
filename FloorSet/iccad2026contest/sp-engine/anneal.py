"""
Simulated annealing over the sequence pair, using the numba JIT energy.

State is P, N (Gamma+/Gamma- orders) with inverse-position arrays posp, posn.
Moves are O(1) swaps in place, reverted on reject.

Shape moves (enable_rotation=True): for free singleton units (rot_ok=True),
continuously vary the aspect ratio while preserving area exactly.  Each shape
move picks one of 7 discrete ratios w:h ∈ {1:3,1:2,2:3,1:1,3:2,2:1,3:1}.
shape_r[u] tracks the current w/h ratio; area_u[u] is the fixed target area.
The returned shape_r can be applied to comp before finalize() in the worker.
"""

import math
import random
import time
from collections import defaultdict

import numpy as np

from fastcore import extract_arrays, energy_nb

W_RES = 25.0
SHAPE_RATIOS = [1/3, 1/2, 2/3, 1.0, 3/2, 2.0, 3.0]   # w:h ratio candidates


def edge_pack_cleanup(P_in, N_in, comp):
    """Apply all edge_pack moves as a post-SA cleanup pass.

    Rearranges the SP so boundary-constrained blocks are placed at their
    required edges (L→x=0, R→x=xmax, B→y=0, T→y=ymax).  Returns the
    modified (P, N) lists; inputs are not modified.
    """
    U = comp.U
    P = list(P_in); N = list(N_in)
    posp = np.empty(U, dtype=np.int64); posp[P] = np.arange(U)
    posn = np.empty(U, dtype=np.int64); posn[N] = np.arange(U)

    ucode = np.zeros(U, dtype=np.int64)
    for u in range(U):
        code = 0
        for (b, _, _) in comp.members[u]:
            code |= int(comp.codes[b])
        ucode[u] = code

    left_u  = [u for u in range(U) if ucode[u] & 1]
    right_u = [u for u in range(U) if ucode[u] & 2]
    top_u   = [u for u in range(U) if ucode[u] & 4]
    bot_u   = [u for u in range(U) if ucode[u] & 8]

    def _pack(edge):
        if edge == 'L' and left_u:
            grp = sorted(left_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = grp[::-1] + [u for u in N if u not in sset]; posn[N] = np.arange(U)
            new_grp_P = sorted(grp, key=lambda u: posn[u], reverse=True)
            P[:] = new_grp_P + [u for u in P if u not in sset]; posp[P] = np.arange(U)
        elif edge == 'R' and right_u:
            grp = sorted(right_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = [u for u in N if u not in sset] + grp[::-1]; posn[N] = np.arange(U)
            new_grp_P = sorted(grp, key=lambda u: posn[u], reverse=True)
            P[:] = [u for u in P if u not in sset] + new_grp_P; posp[P] = np.arange(U)
        elif edge == 'B' and bot_u:
            grp = sorted(bot_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = grp + [u for u in P if u not in sset]; posp[P] = np.arange(U)
        elif edge == 'T' and top_u:
            grp = sorted(top_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = [u for u in P if u not in sset] + grp[::-1]; posp[P] = np.arange(U)

    for edge in ('L', 'R', 'B', 'T'):
        _pack(edge)
    return P, N


def _inv(order, U):
    p = np.empty(U, dtype=np.int64)
    p[order] = np.arange(U, dtype=np.int64)
    return p


def _swap(arr, pos, a, b):
    arr[a], arr[b] = arr[b], arr[a]
    pos[arr[a]] = a
    pos[arr[b]] = b


def anneal(comp, P0, N0, time_budget=2.0, T0=0.03, cool=0.985,
           moves_per_T=None, seed=0, outline=None, w_out=1.5,
           enable_rotation=True):
    """Return (P, N, cost, feas, shape_r).

    shape_r is a float[U] array of final w/h ratios for each unit.
    For units with rot_ok=False, shape_r[u] = w/h (unchanged from initial).
    The caller must apply shape_r to comp (via area_u = comp.uw*comp.uh)
    before calling finalize().
    """
    U = comp.U
    init_shape_r = np.where(comp.uh > 1e-12,
                            comp.uw / (comp.uh + 1e-12), 1.0)
    empty_shape_r = init_shape_r.copy()
    if U <= 1:
        return list(P0), list(N0), 0.0, True, empty_shape_r

    if getattr(comp, "_fast", None) is None:
        comp._fast = extract_arrays(comp)
    A = comp._fast
    # fixed target area per unit (before any shape changes)
    area_u = np.array([float(A["uw"][u]) * float(A["uh"][u]) for u in range(U)])

    charlen = max(math.sqrt(max(comp.area_base, 1e-9)), 1e-6)
    if outline is None:
        wstar = 1e18; hstar = 1e18; wo = 0.0
    else:
        wstar, hstar = float(outline[0]), float(outline[1]); wo = w_out

    rng = random.Random(seed)
    if moves_per_T is None:
        moves_per_T = max(40, 6 * U)

    P = np.asarray(P0, np.int64).copy()
    N = np.asarray(N0, np.int64).copy()
    posp = _inv(P, U)
    posn = _inv(N, U)

    # boundary metadata
    bnd_units = []
    ucode = np.zeros(U, dtype=np.int64)
    for u in range(U):
        code = 0
        for (b, _, _) in comp.members[u]:
            code |= int(comp.codes[b])
        ucode[u] = code
        if code:
            bnd_units.append((u, code))
    left_u  = [u for u in range(U) if ucode[u] & 1]
    right_u = [u for u in range(U) if ucode[u] & 2]
    top_u   = [u for u in range(U) if ucode[u] & 4]
    bot_u   = [u for u in range(U) if ucode[u] & 8]

    cu = defaultdict(set)
    if comp.clu is not None and comp.clu.size:
        for b in range(comp.n):
            c = int(comp.clu[b])
            if c > 0:
                cu[c].add(int(comp.block_unit[b]))
    cluster_list = [sorted(us) for us in cu.values() if len(us) >= 2]

    # ---- shape optimization setup -------------------------------------------
    rot_units_list = []
    rot_blk = {}
    shape_r = init_shape_r.copy()

    if enable_rotation:
        for u in range(U):
            if comp.rot_ok[u] and len(comp.members[u]) == 1:
                rot_units_list.append(u)
                rot_blk[u] = comp.members[u][0][0]

        def _set_shape(u, new_r):
            a = area_u[u]
            new_w = math.sqrt(a * new_r); new_h = math.sqrt(a / new_r)
            A["uw"][u] = new_w; A["uh"][u] = new_h
            b = rot_blk[u]
            A["bw"][b] = new_w; A["bh"][b] = new_h
            shape_r[u] = new_r

        # per-worker random initial shapes for diversity
        rng_init = random.Random(seed ^ 0xA5B6C7D8)
        for u in rot_units_list:
            r0 = rng_init.choice(SHAPE_RATIOS)
            _set_shape(u, r0)

        def restore_shape_to(target_r):
            for u in rot_units_list:
                if abs(shape_r[u] - target_r[u]) > 1e-12:
                    _set_shape(u, float(target_r[u]))
    else:
        def _set_shape(u, new_r):
            pass
        def restore_shape_to(target_r):
            pass
    # -------------------------------------------------------------------------

    best_shape_r = shape_r.copy()
    best_feas_shape_r = shape_r.copy()

    def edge_pack(edge):
        if edge == 'L' and left_u:
            # LEFT blocks form a vertical column at x=0.
            # N: LEFT blocks at positions 0..k-1 (any order).
            # P: LEFT blocks in DESCENDING N order at start of P.
            #    With this, no LEFT block has a LEFT-of-it predecessor → all get x=0.
            grp = sorted(left_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = grp[::-1] + [u for u in N if u not in sset]; posn[N] = np.arange(U)
            new_grp_P = sorted(grp, key=lambda u: posn[u], reverse=True)
            P[:] = new_grp_P + [u for u in P if u not in sset]; posp[P] = np.arange(U)
        elif edge == 'R' and right_u:
            # RIGHT blocks: N at end, P in descending N order at end of P.
            grp = sorted(right_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = [u for u in N if u not in sset] + grp[::-1]; posn[N] = np.arange(U)
            new_grp_P = sorted(grp, key=lambda u: posn[u], reverse=True)
            P[:] = [u for u in P if u not in sset] + new_grp_P; posp[P] = np.arange(U)
        elif edge == 'B' and bot_u:
            # BOTTOM blocks form a horizontal row at y=0.
            # P: BOTTOM blocks in ASCENDING N order at start of P (NOT reversed).
            #    Each predecessor in P has smaller N-pos → is LEFT of the current block, not BELOW.
            #    So uy=0 for every BOTTOM block. N unchanged to preserve layout continuity.
            grp = sorted(bot_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = grp + [u for u in P if u not in sset]; posp[P] = np.arange(U)
        elif edge == 'T' and top_u:
            # TOP blocks: put in descending N order at END of P.
            # The block with the smallest N-pos ends up last in P → all other blocks
            # precede it and those with larger N-pos are BELOW it → gets max y.
            grp = sorted(top_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = [u for u in P if u not in sset] + grp[::-1]; posp[P] = np.arange(U)

    def cluster_pack(group):
        sset = set(group)
        gP = sorted(group, key=lambda u: posp[u])
        restP = [u for u in P if u not in sset]
        aP = min(len(restP), int(np.median([posp[u] for u in group])))
        P[:] = restP[:aP] + gP + restP[aP:]; posp[P] = np.arange(U)
        gN = sorted(group, key=lambda u: posn[u])
        restN = [u for u in N if u not in sset]
        aN = min(len(restN), int(np.median([posn[u] for u in group])))
        N[:] = restN[:aN] + gN + restN[aN:]; posn[N] = np.arange(U)

    def ev():
        return energy_nb(P, posn, A["uw"], A["uh"], A["pre"], A["px"], A["py"],
                         A["block_unit"], A["offx"], A["offy"], A["bw"], A["bh"],
                         A["codes"], A["b2b_i"], A["b2b_j"], A["b2b_w"],
                         A["p2b_pin"], A["p2b_blk"], A["p2b_w"], A["pin_x"],
                         A["pin_y"], A["hpwl_base"], A["area_base"], A["n_soft"],
                         W_RES, charlen, A["cmem"], A["cptr"], wstar, hstar, wo)

    def gen_swaps():
        r = rng.random()
        if bnd_units and r < 0.18:
            u, code = bnd_units[rng.randrange(len(bnd_units))]
            sw = []
            if code & 1:  sw.append((1, posn[u], 0))
            if code & 2:  sw.append((1, posn[u], U - 1))
            if code & 8:  sw.append((0, posp[u], 0))
            if code & 4:  sw.append((0, posp[u], U - 1))
            return [s for s in sw if s[1] != s[2]]
        if r < 0.45:
            return [(0, rng.randrange(U), rng.randrange(U))]
        if r < 0.70:
            return [(1, rng.randrange(U), rng.randrange(U))]
        u = rng.randrange(U); v = rng.randrange(U)
        if u == v:
            return [(0, rng.randrange(U), rng.randrange(U))]
        return [(0, posp[u], posp[v]), (1, posn[u], posn[v])]

    def apply(sw):
        for tag, i, j in sw:
            _swap(P, posp, i, j) if tag == 0 else _swap(N, posn, i, j)

    def undo(sw):
        for tag, i, j in reversed(sw):
            _swap(P, posp, i, j) if tag == 0 else _swap(N, posn, i, j)

    edges = [e for e, lst in (('L', left_u), ('R', right_u),
                               ('B', bot_u), ('T', top_u)) if lst]

    def restore_to(state_pn, state_shape=None):
        P[:] = state_pn[0]; posp[P] = np.arange(U)
        N[:] = state_pn[1]; posn[N] = np.arange(U)
        if state_shape is not None:
            restore_shape_to(state_shape)

    e, cost, res, _ = ev()
    best_P = P.copy(); best_N = N.copy()
    best_feas_cost = cost if res < 1e-6 else float("inf")
    best_feas = (P.copy(), N.copy()) if res < 1e-6 else None
    best_e = e

    t0 = time.time()
    reheats = 0
    while time.time() - t0 < time_budget:
        T = T0 if reheats == 0 else T0 * 0.6
        while T > 1e-4:
            for _ in range(moves_per_T):
                r = rng.random()
                is_big = False
                shape_move = False
                shape_u = -1; old_r_val = 1.0

                if r < 0.10 and edges:
                    is_big = True
                    snapP = P.copy(); snapN = N.copy()
                    edge_pack(edges[rng.randrange(len(edges))])
                elif r < 0.18 and cluster_list:
                    is_big = True
                    snapP = P.copy(); snapN = N.copy()
                    cluster_pack(cluster_list[rng.randrange(len(cluster_list))])
                elif r < 0.30 and rot_units_list:
                    shape_move = True
                    shape_u = rot_units_list[rng.randrange(len(rot_units_list))]
                    old_r_val = shape_r[shape_u]
                    new_r_val = SHAPE_RATIOS[rng.randrange(len(SHAPE_RATIOS))]
                    _set_shape(shape_u, new_r_val)
                else:
                    sw = gen_swaps()
                    if not sw:
                        continue
                    apply(sw)

                ne, ncost, nres, _ = ev()
                d = ne - e
                if d < 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
                    e, cost, res = ne, ncost, nres
                    if e < best_e:
                        best_e = e
                        best_P = P.copy(); best_N = N.copy()
                        best_shape_r = shape_r.copy()
                    if res < 1e-6 and cost < best_feas_cost:
                        best_feas_cost = cost
                        best_feas = (P.copy(), N.copy())
                        best_feas_shape_r = shape_r.copy()
                else:
                    if is_big:
                        P[:] = snapP; posp[P] = np.arange(U)
                        N[:] = snapN; posn[N] = np.arange(U)
                    elif shape_move:
                        _set_shape(shape_u, old_r_val)   # undo
                    else:
                        undo(sw)

            T *= cool
            if time.time() - t0 >= time_budget:
                break

        if best_feas is not None:
            restore_to(best_feas, best_feas_shape_r)
        else:
            restore_to((best_P, best_N), best_shape_r)
        e, cost, res, _ = ev()
        reheats += 1

    if best_feas is not None:
        return list(best_feas[0]), list(best_feas[1]), best_feas_cost, True, best_feas_shape_r
    return list(best_P), list(best_N), best_e, False, best_shape_r
