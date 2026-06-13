"""
Simulated annealing over the sequence pair, using the numba JIT energy.

State is held as numpy arrays P, N (Gamma+/Gamma- orders) with inverse-position
arrays posp, posn. Moves are O(1) swaps applied in place and reverted on reject
(no list copies), so throughput is bound by the JIT energy, not Python overhead.

Energy = exact contest quality cost on the compacted pack + residual penalty for
preplaced infeasibility. Slack wirelength redistribution is applied once to the
incumbent in floorplanner.finalize.
"""

import math
import random
import time

import numpy as np

from fastcore import extract_arrays, energy_nb

W_RES = 25.0


def _inv(order, U):
    p = np.empty(U, dtype=np.int64)
    p[order] = np.arange(U, dtype=np.int64)
    return p


def _swap(arr, pos, a, b):
    arr[a], arr[b] = arr[b], arr[a]
    pos[arr[a]] = a
    pos[arr[b]] = b


def anneal(comp, P0, N0, time_budget=2.0, T0=0.03, cool=0.985,
           moves_per_T=None, seed=0, outline=None, w_out=1.5):
    U = comp.U
    if U <= 1:
        return list(P0), list(N0), 0.0, True

    if getattr(comp, "_fast", None) is None:
        comp._fast = extract_arrays(comp)
    A = comp._fast
    charlen = max(math.sqrt(max(comp.area_base, 1e-9)), 1e-6)
    # target outline (W*, H*); no penalty if not provided
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

    bnd_units = []
    ucode = np.zeros(U, dtype=np.int64)
    for u in range(U):
        code = 0
        for (b, _, _) in comp.members[u]:
            code |= int(comp.codes[b])
        ucode[u] = code
        if code:
            bnd_units.append((u, code))
    left_u = [u for u in range(U) if ucode[u] & 1]
    right_u = [u for u in range(U) if ucode[u] & 2]
    top_u = [u for u in range(U) if ucode[u] & 4]
    bot_u = [u for u in range(U) if ucode[u] & 8]

    # cluster -> member units (macros off: each member is its own unit)
    cluster_units = {}
    if comp.clu is not None and comp.clu.size:
        from collections import defaultdict
        cu = defaultdict(set)
        for b in range(comp.n):
            c = int(comp.clu[b])
            if c > 0:
                cu[c].add(int(comp.block_unit[b]))
        cluster_units = {c: sorted(us) for c, us in cu.items() if len(us) >= 2}
    cluster_list = list(cluster_units.values())

    def edge_pack(edge):
        """Put all of one edge's units at the sequence extreme, stacked (reverse
        order in the opposite sequence) so they all touch that edge."""
        if edge == 'L' and left_u:
            grp = sorted(left_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = grp[::-1] + [u for u in N if u not in sset]; posn[N] = np.arange(U)
        elif edge == 'R' and right_u:
            grp = sorted(right_u, key=lambda u: posp[u]); sset = set(grp)
            N[:] = [u for u in N if u not in sset] + grp[::-1]; posn[N] = np.arange(U)
        elif edge == 'B' and bot_u:
            grp = sorted(bot_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = grp[::-1] + [u for u in P if u not in sset]; posp[P] = np.arange(U)
        elif edge == 'T' and top_u:
            grp = sorted(top_u, key=lambda u: posn[u]); sset = set(grp)
            P[:] = [u for u in P if u not in sset] + grp[::-1]; posp[P] = np.arange(U)

    def cluster_pack(group):
        """Make a cluster's units contiguous in both sequences (preserving their
        relative order) so they pack as one spatial region and can abut."""
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

    e, cost, res, _ = ev()
    best_P = P.copy(); best_N = N.copy()
    best_feas_cost = cost if res < 1e-6 else float("inf")
    best_feas = (P.copy(), N.copy()) if res < 1e-6 else None
    best_e = e

    def gen_swaps():
        """Return a list of (array_tag, i, j) swaps. tag 0=P, 1=N."""
        r = rng.random()
        if bnd_units and r < 0.18:
            u, code = bnd_units[rng.randrange(len(bnd_units))]
            sw = []
            if code & 1:
                sw.append((1, posn[u], 0))
            if code & 2:
                sw.append((1, posn[u], U - 1))
            if code & 8:
                sw.append((0, posp[u], 0))
            if code & 4:
                sw.append((0, posp[u], U - 1))
            return [s for s in sw if s[1] != s[2]]
        if r < 0.45:
            return [(0, rng.randrange(U), rng.randrange(U))]
        if r < 0.70:
            return [(1, rng.randrange(U), rng.randrange(U))]
        # double swap: same two units in both sequences
        u = rng.randrange(U); v = rng.randrange(U)
        if u == v:
            return [(0, rng.randrange(U), rng.randrange(U))]
        return [(0, posp[u], posp[v]), (1, posn[u], posn[v])]

    def apply(sw):
        for tag, i, j in sw:
            if tag == 0:
                _swap(P, posp, i, j)
            else:
                _swap(N, posn, i, j)

    def undo(sw):
        for tag, i, j in reversed(sw):
            if tag == 0:
                _swap(P, posp, i, j)
            else:
                _swap(N, posn, i, j)

    edges = [e for e, lst in (('L', left_u), ('R', right_u),
                              ('B', bot_u), ('T', top_u)) if lst]

    def restore_to(state):
        P[:] = state[0]; posp[P] = np.arange(U)
        N[:] = state[1]; posn[N] = np.arange(U)

    t0 = time.time()
    it = 0
    reheats = 0
    # Outer reheating loop: each pass cools from T0; on cool-out we restart from
    # the best incumbent and reheat, using the whole budget to escape local minima.
    while time.time() - t0 < time_budget:
        T = T0 if reheats == 0 else T0 * 0.6
        while T > 1e-4:
            for _ in range(moves_per_T):
                r = rng.random()
                big = (r < 0.10 and edges) or (r < 0.18 and cluster_list)
                if big:
                    snapP = P.copy(); snapN = N.copy()
                    if r < 0.10:
                        edge_pack(edges[rng.randrange(len(edges))])
                    else:
                        cluster_pack(cluster_list[rng.randrange(len(cluster_list))])
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
                        best_e = e; best_P = P.copy(); best_N = N.copy()
                    if res < 1e-6 and cost < best_feas_cost:
                        best_feas_cost = cost
                        best_feas = (P.copy(), N.copy())
                else:
                    if big:
                        P[:] = snapP; posp[P] = np.arange(U)
                        N[:] = snapN; posn[N] = np.arange(U)
                    else:
                        undo(sw)
                it += 1
            T *= cool
            if time.time() - t0 >= time_budget:
                break
        # reheat from the best incumbent so far
        restore_to(best_feas if best_feas is not None else (best_P, best_N))
        e, cost, res, _ = ev()
        reheats += 1

    if best_feas is not None:
        return list(best_feas[0]), list(best_feas[1]), best_feas_cost, True
    return list(best_P), list(best_N), best_e, False
