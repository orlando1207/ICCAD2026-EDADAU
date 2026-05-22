#!/usr/bin/env python3
"""
Constraint-aware B*-tree simulated annealing baseline for FloorSet Lite.

Core design:
  - Original block IDs are preserved through the whole flow.
  - Preplaced blocks are immutable obstacles and are excluded from B*-trees.
  - Soft-block dimensions are derived from area and aspect ratio.
  - MIB soft groups share one aspect-ratio state and one average group area.
  - Cluster/grouping constraints use hierarchical sub-B*-trees at high
    temperature and are de-clustered into individual blocks at low temperature.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (  # noqa: E402
    FloorplanOptimizer,
    calculate_bbox_area,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
)


EPS = 1e-6
AR_MIN = 0.10
AR_MAX = 10.0
INF = 1.0e100


@dataclass
class BlockInfo:
    block_id: int
    area: float
    is_fixed: bool
    is_preplaced: bool
    mib: int
    cluster: int
    boundary: int
    target_x: float = -1.0
    target_y: float = -1.0
    target_w: float = -1.0
    target_h: float = -1.0


@dataclass
class PackedItem:
    x: float
    y: float
    w: float
    h: float


class ShapeState:
    """Owns all mutable shape variables.

    Soft blocks use AR-derived dimensions. Soft MIB groups share a single AR
    and the group's average area, matching the requested by-construction MIB
    model. Fixed and preplaced dimensions are immutable for evaluator safety.
    """

    def __init__(self, blocks: Dict[int, BlockInfo]):
        self.blocks = blocks
        self.block_ar: Dict[int, float] = {}
        self.mib_ar: Dict[int, float] = {}
        self.mib_area: Dict[int, float] = {}

        mib_members: Dict[int, List[int]] = {}
        for block in blocks.values():
            if block.is_fixed or block.is_preplaced:
                continue
            if block.mib > 0:
                mib_members.setdefault(block.mib, []).append(block.block_id)
            else:
                self.block_ar[block.block_id] = 1.0

        for mib, members in mib_members.items():
            self.mib_ar[mib] = 1.0
            self.mib_area[mib] = sum(blocks[i].area for i in members) / max(1, len(members))

    def copy(self) -> "ShapeState":
        new = ShapeState.__new__(ShapeState)
        new.blocks = self.blocks
        new.block_ar = self.block_ar.copy()
        new.mib_ar = self.mib_ar.copy()
        new.mib_area = self.mib_area.copy()
        return new

    def _soft_area_ar(self, block_id: int) -> Tuple[float, float]:
        block = self.blocks[block_id]
        if block.mib > 0 and block.mib in self.mib_ar:
            return self.mib_area[block.mib], self.mib_ar[block.mib]
        return block.area, self.block_ar.get(block_id, 1.0)

    def dimensions(self, block_id: int) -> Tuple[float, float]:
        block = self.blocks[block_id]
        if block.is_fixed or block.is_preplaced:
            return block.target_w, block.target_h

        area, ar = self._soft_area_ar(block_id)
        ar = min(AR_MAX, max(AR_MIN, ar))
        return math.sqrt(max(area, EPS) * ar), math.sqrt(max(area, EPS) / ar)

    def soft_shape_keys(self) -> List[Tuple[str, int]]:
        keys: List[Tuple[str, int]] = [("block", i) for i in self.block_ar]
        keys.extend(("mib", g) for g in self.mib_ar)
        return keys

    def rotate_key(self, key: Tuple[str, int]) -> None:
        kind, idx = key
        if kind == "mib":
            self.mib_ar[idx] = self._clamp_ar(1.0 / max(self.mib_ar[idx], EPS))
        else:
            self.block_ar[idx] = self._clamp_ar(1.0 / max(self.block_ar[idx], EPS))

    def reshape_key(self, key: Tuple[str, int], delta: float, divide: bool) -> None:
        kind, idx = key
        if divide:
            factor = 1.0 / max(0.05, 1.0 - delta)
        else:
            factor = 1.0 + delta
        if random.random() < 0.5:
            factor = 1.0 / factor

        if kind == "mib":
            self.mib_ar[idx] = self._clamp_ar(self.mib_ar[idx] * factor)
        else:
            self.block_ar[idx] = self._clamp_ar(self.block_ar[idx] * factor)

    @staticmethod
    def _clamp_ar(ar: float) -> float:
        return min(AR_MAX, max(AR_MIN, ar))


class BStarTree:
    """B*-tree over arbitrary payload IDs.

    Left child means right-adjacent in the packed layout. Right child means
    top-adjacent with the same x-coordinate, with the contour choosing y.
    """

    def __init__(self, payloads: Sequence[int], deterministic: bool = False):
        self.payloads = list(payloads)
        self.n = len(self.payloads)
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        self.root = 0 if self.n else -1
        if deterministic:
            self._build_chain()
        else:
            self._build_random()

    def copy(self) -> "BStarTree":
        new = BStarTree.__new__(BStarTree)
        new.payloads = self.payloads.copy()
        new.n = self.n
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[int]]) -> "BStarTree":
        """Build a compact shelf-like B*-tree.

        The first item in each row is connected through right children, and
        each row is a left-child chain. Under the packer's semantics this gives
        a dense multi-row starting point instead of the long one-row chain.
        """
        payloads = [payload for row in rows for payload in row]
        tree = cls.__new__(cls)
        tree.payloads = payloads
        tree.n = len(payloads)
        tree.parent = [-1] * tree.n
        tree.left = [-1] * tree.n
        tree.right = [-1] * tree.n
        tree.root = 0 if tree.n else -1

        offset = 0
        previous_row_head = -1
        for row in rows:
            if not row:
                continue
            row_head = offset
            if previous_row_head != -1:
                tree.right[previous_row_head] = row_head
                tree.parent[row_head] = previous_row_head
            for idx in range(len(row) - 1):
                tree.left[offset + idx] = offset + idx + 1
                tree.parent[offset + idx + 1] = offset + idx
            previous_row_head = row_head
            offset += len(row)
        return tree

    def _build_chain(self) -> None:
        for i in range(1, self.n):
            self.left[i - 1] = i
            self.parent[i] = i - 1

    def _build_random(self) -> None:
        if self.n <= 1:
            return
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        placed = [self.root]
        for node in order[1:]:
            while True:
                target = random.choice(placed)
                side = random.randint(0, 1)
                child = self.left[target] if side == 0 else self.right[target]
                if child == -1:
                    if side == 0:
                        self.left[target] = node
                    else:
                        self.right[target] = node
                    self.parent[node] = target
                    placed.append(node)
                    break
                placed.append(child)

    def payload_order(self) -> List[int]:
        out: List[int] = []

        def dfs(node: int) -> None:
            if node == -1:
                return
            out.append(self.payloads[node])
            dfs(self.left[node])
            dfs(self.right[node])

        dfs(self.root)
        return out

    def move_swap_payloads(self) -> None:
        if self.n < 2:
            return
        a, b = random.sample(range(self.n), 2)
        self.payloads[a], self.payloads[b] = self.payloads[b], self.payloads[a]

    def move_delete_insert(self) -> None:
        if self.n < 2:
            return
        node = random.randrange(self.n)
        self._delete_node(node)
        candidates = [i for i in range(self.n) if i != node]
        if not candidates:
            return
        target = random.choice(candidates)
        self._insert_node(node, target, random.choice([True, False]))

    def _delete_node(self, node: int) -> None:
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]

        if left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            cursor = left_child
            while self.right[cursor] != -1:
                cursor = self.right[cursor]
            self.right[cursor] = right_child
            self.parent[right_child] = cursor

        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement

        if replacement != -1:
            self.parent[replacement] = parent

        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1

    def _insert_node(self, node: int, target: int, as_left: bool) -> None:
        self.parent[node] = target
        self.left[node] = -1
        self.right[node] = -1
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
            self.left[node] = old_child
        else:
            old_child = self.right[target]
            self.right[target] = node
            self.right[node] = old_child
        if old_child != -1:
            self.parent[old_child] = node

    def pack(
        self,
        dim_fn,
        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Dict[int, PackedItem]:
        if self.n == 0:
            return {}

        contour = Contour()
        obstacles = obstacles or []
        for ox, oy, ow, oh in obstacles:
            contour.update(ox, ox + ow, oy + oh)

        packed: Dict[int, PackedItem] = {}

        def clear_obstacles(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
            for _ in range(64):
                y = max(y, contour.max_y(x, x + w))
                new_y = y
                for ox, oy, ow, oh in obstacles:
                    if rects_overlap(x, y, w, h, ox, oy, ow, oh):
                        new_y = max(new_y, oy + oh)
                if new_y <= y + EPS:
                    return x, y
                y = new_y
            return x, y

        def dfs(node: int, anchor_x: float) -> None:
            if node == -1:
                return
            payload = self.payloads[node]
            w, h = dim_fn(payload)
            x = 0.0 if node == self.root else anchor_x
            y = contour.max_y(x, x + w)
            x, y = clear_obstacles(x, y, w, h)
            packed[payload] = PackedItem(x, y, w, h)
            contour.update(x, x + w, y + h)
            dfs(self.left[node], x + w)
            dfs(self.right[node], x)

        dfs(self.root, 0.0)
        return packed


class Contour:
    """Simple 1D skyline as disjoint x-intervals with top y."""

    def __init__(self) -> None:
        self.segments: List[Tuple[float, float, float]] = []

    def max_y(self, x0: float, x1: float) -> float:
        max_y = 0.0
        for sx0, sx1, sy in self.segments:
            if x0 < sx1 - EPS and x1 > sx0 + EPS:
                max_y = max(max_y, sy)
        return max_y

    def update(self, x0: float, x1: float, y_top: float) -> None:
        if x1 <= x0 + EPS:
            return
        new_segments: List[Tuple[float, float, float]] = []
        for sx0, sx1, sy in self.segments:
            if sx1 <= x0 + EPS or sx0 >= x1 - EPS:
                new_segments.append((sx0, sx1, sy))
                continue
            if sx0 < x0 - EPS:
                new_segments.append((sx0, x0, sy))
            if sx1 > x1 + EPS:
                new_segments.append((x1, sx1, sy))
        new_segments.append((x0, x1, y_top))
        new_segments.sort(key=lambda s: (s[0], s[1]))

        merged: List[Tuple[float, float, float]] = []
        for sx0, sx1, sy in new_segments:
            if not merged:
                merged.append((sx0, sx1, sy))
            else:
                px0, px1, py = merged[-1]
                if abs(px1 - sx0) < EPS and abs(py - sy) < EPS:
                    merged[-1] = (px0, sx1, py)
                else:
                    merged.append((sx0, sx1, sy))
        self.segments = merged


@dataclass
class GroupMacro:
    group_id: int
    members: List[int]
    tree: BStarTree

    def copy(self) -> "GroupMacro":
        return GroupMacro(self.group_id, self.members.copy(), self.tree.copy())


@dataclass
class SearchState:
    tree: BStarTree
    shape: ShapeState
    groups: Dict[int, GroupMacro]

    def copy(self) -> "SearchState":
        return SearchState(
            self.tree.copy(),
            self.shape.copy(),
            {gid: group.copy() for gid, group in self.groups.items()},
        )


def rects_overlap(
    x1: float, y1: float, w1: float, h1: float,
    x2: float, y2: float, w2: float, h2: float,
) -> bool:
    return (
        min(x1 + w1, x2 + w2) - max(x1, x2) > EPS
        and min(y1 + h1, y2 + h2) - max(y1, y2) > EPS
    )


class MyOptimizer(FloorplanOptimizer):
    """Hierarchical B*-tree floorplanner with two-stage SA."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.high_temps = 1
        self.low_temps = 1
        self.cooling_rate = 0.88
        self.moves_factor = 2

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:
        blocks = self._parse_blocks(block_count, area_targets, constraints, target_positions)
        preplaced = self._preplaced_positions(blocks)
        movable = [i for i, b in blocks.items() if not b.is_preplaced]
        shape = ShapeState(blocks)

        if not movable:
            return [preplaced[i] for i in range(block_count)]

        groups = self._build_group_macros(blocks, movable)
        grouped_members = {bid for group in groups.values() for bid in group.members}
        high_payloads: List[int] = []
        for gid in sorted(groups):
            high_payloads.append(-gid)
        high_payloads.extend(i for i in movable if i not in grouped_members)
        high_payloads = self._ordered_payloads(high_payloads, blocks, groups)

        high_tree = self._build_shelf_tree(
            high_payloads, shape, groups, b2b_connectivity, p2b_connectivity, pins_pos
        )
        high_state = SearchState(high_tree, shape, groups)
        high_state = self._anneal(
            high_state, preplaced, b2b_connectivity, p2b_connectivity, pins_pos,
            blocks, phase="high", temp_steps=self.high_temps,
        )
        high_positions = self._pack_state(high_state, preplaced, phase="high")

        low_payloads = self._refine_low_order(movable, high_positions, b2b_connectivity, p2b_connectivity)
        low_tree = self._build_shelf_tree(
            low_payloads, high_state.shape, {}, b2b_connectivity, p2b_connectivity, pins_pos
        )
        low_state = SearchState(low_tree, high_state.shape, {})
        low_state = self._anneal(
            low_state, preplaced, b2b_connectivity, p2b_connectivity, pins_pos,
            blocks, phase="low", temp_steps=self.low_temps,
        )
        best_positions = self._pack_state(low_state, preplaced, phase="low")

        if not math.isfinite(self._cost(best_positions, b2b_connectivity, p2b_connectivity, pins_pos, blocks, "low")):
            return self._fallback_shelf(block_count, blocks)
        return best_positions

    def _parse_blocks(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> Dict[int, BlockInfo]:
        blocks: Dict[int, BlockInfo] = {}
        ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
        for i in range(block_count):
            area = float(area_targets[i]) if i < len(area_targets) and area_targets[i] > 0 else 1.0
            is_fixed = ncols > 0 and bool(constraints[i, 0] != 0)
            is_preplaced = ncols > 1 and bool(constraints[i, 1] != 0)
            mib = int(constraints[i, 2].item()) if ncols > 2 else 0
            cluster = int(constraints[i, 3].item()) if ncols > 3 else 0
            boundary = int(constraints[i, 4].item()) if ncols > 4 else 0

            tx = ty = tw = th = -1.0
            if target_positions is not None and i < len(target_positions):
                tx, ty, tw, th = [float(v) for v in target_positions[i]]
            if (is_fixed or is_preplaced) and (tw <= 0.0 or th <= 0.0):
                side = math.sqrt(max(area, EPS))
                tw, th = side, side
            if is_preplaced and (tx < 0.0 or ty < 0.0):
                tx, ty = 0.0, 0.0

            blocks[i] = BlockInfo(
                block_id=i,
                area=area,
                is_fixed=is_fixed,
                is_preplaced=is_preplaced,
                mib=mib,
                cluster=cluster,
                boundary=boundary,
                target_x=tx,
                target_y=ty,
                target_w=tw,
                target_h=th,
            )
        return blocks

    def _preplaced_positions(self, blocks: Dict[int, BlockInfo]) -> Dict[int, Tuple[float, float, float, float]]:
        out: Dict[int, Tuple[float, float, float, float]] = {}
        for bid, block in blocks.items():
            if block.is_preplaced:
                out[bid] = (block.target_x, block.target_y, block.target_w, block.target_h)
        return out

    def _build_group_macros(self, blocks: Dict[int, BlockInfo], movable: Iterable[int]) -> Dict[int, GroupMacro]:
        by_group: Dict[int, List[int]] = {}
        movable_set = set(movable)
        for bid in movable_set:
            group_id = blocks[bid].cluster
            if group_id > 0:
                by_group.setdefault(group_id, []).append(bid)

        groups: Dict[int, GroupMacro] = {}
        for gid, members in by_group.items():
            if len(members) < 2:
                continue
            members.sort(key=lambda bid: (-blocks[bid].area, bid))
            tree = BStarTree(members, deterministic=True)
            groups[gid] = GroupMacro(gid, members, tree)
        return groups

    def _ordered_payloads(
        self,
        payloads: List[int],
        blocks: Dict[int, BlockInfo],
        groups: Dict[int, GroupMacro],
    ) -> List[int]:
        def key(payload: int) -> Tuple[int, float, int]:
            if payload < 0:
                members = groups[-payload].members
                boundary = max(blocks[i].boundary for i in members)
                area = sum(blocks[i].area for i in members)
                return (0 if boundary else 1, -area, payload)
            block = blocks[payload]
            return (0 if block.boundary else 1, -block.area, payload)

        return sorted(payloads, key=key)

    def _build_shelf_tree(
        self,
        payloads: List[int],
        shape: ShapeState,
        groups: Dict[int, GroupMacro],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
    ) -> BStarTree:
        if not payloads:
            return BStarTree([])

        dims = {payload: self._estimate_payload_dims(payload, shape, groups) for payload in payloads}
        total_area = sum(w * h for w, h in dims.values())
        target_width = self._target_shelf_width(payloads, dims)
        ordered = self._connectivity_order(payloads, shape.blocks, groups, b2b, p2b, pins)

        rows: List[List[int]] = []
        current: List[int] = []
        current_width = 0.0
        for payload in ordered:
            w, _ = dims[payload]
            over_width = current and current_width + w > target_width
            enough_rows = len(rows) < max(1, int(math.sqrt(len(payloads))) + 1)
            if over_width and enough_rows and current_width >= 0.55 * target_width:
                rows.append(current)
                current = []
                current_width = 0.0
            current.append(payload)
            current_width += w
        if current:
            rows.append(current)

        if len(rows) == 1 and len(payloads) > 4:
            rows = self._rebalance_rows(ordered, dims, max(2, round(math.sqrt(total_area) / max(target_width, EPS)) + 1))
        return BStarTree.from_rows(rows)

    def _estimate_payload_dims(
        self,
        payload: int,
        shape: ShapeState,
        groups: Dict[int, GroupMacro],
    ) -> Tuple[float, float]:
        if payload >= 0:
            return shape.dimensions(payload)

        group = groups[-payload]
        widths = []
        heights = []
        for bid in group.members:
            w, h = shape.dimensions(bid)
            widths.append(w)
            heights.append(h)
        if not widths:
            return 1.0, 1.0

        total_area = sum(w * h for w, h in zip(widths, heights))
        side = math.sqrt(max(total_area, EPS))
        max_w = max(widths)
        max_h = max(heights)
        return max(side, max_w), max(side, max_h)

    def _target_shelf_width(
        self,
        payloads: List[int],
        dims: Dict[int, Tuple[float, float]],
    ) -> float:
        total_area = sum(w * h for w, h in dims.values())
        max_w = max((w for w, h in dims.values()), default=1.0)
        max_h = max((h for w, h in dims.values()), default=1.0)
        side = math.sqrt(max(total_area, EPS))
        row_count = max(2, round(math.sqrt(max(1, len(payloads))) * 0.85))
        width_by_rows = sum(w for w, h in dims.values()) / row_count
        return max(max_w, side * 1.08, width_by_rows, max_h * 1.8)

    def _rebalance_rows(
        self,
        ordered: List[int],
        dims: Dict[int, Tuple[float, float]],
        row_count: int,
    ) -> List[List[int]]:
        row_count = max(1, min(row_count, len(ordered)))
        rows: List[List[int]] = [[] for _ in range(row_count)]
        widths = [0.0] * row_count
        for payload in ordered:
            idx = min(range(row_count), key=lambda i: widths[i])
            rows[idx].append(payload)
            widths[idx] += dims[payload][0]
        return [row for row in rows if row]

    def _connectivity_order(
        self,
        payloads: List[int],
        blocks: Dict[int, BlockInfo],
        groups: Dict[int, GroupMacro],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
    ) -> List[int]:
        payload_set = set(payloads)
        block_to_payload: Dict[int, int] = {}
        for payload in payloads:
            if payload >= 0:
                block_to_payload[payload] = payload
            else:
                for bid in groups[-payload].members:
                    block_to_payload[bid] = payload

        adjacency: Dict[int, Dict[int, float]] = {payload: {} for payload in payloads}
        degree: Dict[int, float] = {payload: 0.0 for payload in payloads}

        if b2b is not None:
            for edge in b2b:
                if edge[0] == -1:
                    continue
                a = block_to_payload.get(int(edge[0]))
                b = block_to_payload.get(int(edge[1]))
                if a is None or b is None or a == b:
                    continue
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
                adjacency[b][a] = adjacency[b].get(a, 0.0) + weight
                degree[a] += weight
                degree[b] += weight

        if p2b is not None:
            for edge in p2b:
                if edge[0] == -1:
                    continue
                payload = block_to_payload.get(int(edge[1]))
                if payload is None:
                    continue
                degree[payload] += float(edge[2]) if len(edge) > 2 else 1.0

        def seed_key(payload: int) -> Tuple[int, float, float, int]:
            area = self._payload_area(payload, blocks, groups)
            boundary = self._payload_boundary(payload, blocks, groups)
            return (0 if boundary else 1, -degree.get(payload, 0.0), -area, payload)

        remaining = set(payload_set)
        ordered: List[int] = []
        while remaining:
            if not ordered:
                current = min(remaining, key=seed_key)
            else:
                previous = ordered[-1]
                neighbors = [p for p in adjacency.get(previous, {}) if p in remaining]
                if neighbors:
                    current = max(neighbors, key=lambda p: (adjacency[previous][p], degree.get(p, 0.0), self._payload_area(p, blocks, groups)))
                else:
                    current = min(remaining, key=seed_key)
            ordered.append(current)
            remaining.remove(current)
        return ordered

    def _refine_low_order(
        self,
        movable: List[int],
        high_positions: List[Tuple[float, float, float, float]],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
    ) -> List[int]:
        order = sorted(movable, key=lambda bid: (high_positions[bid][1], high_positions[bid][0]))
        rank = {bid: idx for idx, bid in enumerate(order)}
        degree = {bid: 0.0 for bid in movable}
        for conn in (b2b, p2b):
            if conn is None:
                continue
            for edge in conn:
                if edge[0] == -1:
                    continue
                if conn is b2b:
                    ids = [int(edge[0]), int(edge[1])]
                else:
                    ids = [int(edge[1])]
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                for bid in ids:
                    if bid in degree:
                        degree[bid] += weight

        return sorted(movable, key=lambda bid: (rank.get(bid, 0) // 8, -degree.get(bid, 0.0), rank.get(bid, 0)))

    def _payload_area(self, payload: int, blocks: Dict[int, BlockInfo], groups: Dict[int, GroupMacro]) -> float:
        if payload >= 0:
            return blocks[payload].area
        return sum(blocks[bid].area for bid in groups[-payload].members)

    def _payload_boundary(self, payload: int, blocks: Dict[int, BlockInfo], groups: Dict[int, GroupMacro]) -> int:
        if payload >= 0:
            return blocks[payload].boundary
        return max((blocks[bid].boundary for bid in groups[-payload].members), default=0)

    def _anneal(
        self,
        state: SearchState,
        preplaced: Dict[int, Tuple[float, float, float, float]],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        blocks: Dict[int, BlockInfo],
        phase: str,
        temp_steps: int,
    ) -> SearchState:
        current = state.copy()
        current_pos = self._pack_state(current, preplaced, phase)
        current_cost = self._cost(current_pos, b2b, p2b, pins, blocks, phase)
        best = current.copy()
        best_cost = current_cost

        temp = self._initial_temperature(current, preplaced, b2b, p2b, pins, blocks, phase)
        moves_per_temp = min(8, max(4, self.moves_factor * max(1, current.tree.n)))

        for _ in range(temp_steps):
            for _ in range(moves_per_temp):
                proposal = current.copy()
                self._perturb(proposal, phase)
                new_pos = self._pack_state(proposal, preplaced, phase)
                new_cost = self._cost(new_pos, b2b, p2b, pins, blocks, phase)
                delta = new_cost - current_cost
                if delta <= 0.0 or (math.isfinite(new_cost) and random.random() < math.exp(-delta / max(temp, 1e-9))):
                    current = proposal
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best = proposal.copy()
                        best_cost = new_cost
            temp *= self.cooling_rate
        return best

    def _initial_temperature(
        self,
        state: SearchState,
        preplaced: Dict[int, Tuple[float, float, float, float]],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        blocks: Dict[int, BlockInfo],
        phase: str,
    ) -> float:
        base_pos = self._pack_state(state, preplaced, phase)
        base_cost = self._cost(base_pos, b2b, p2b, pins, blocks, phase)
        deltas: List[float] = []
        for _ in range(3):
            trial = state.copy()
            self._perturb(trial, phase)
            cost = self._cost(self._pack_state(trial, preplaced, phase), b2b, p2b, pins, blocks, phase)
            if math.isfinite(cost) and math.isfinite(base_cost):
                deltas.append(abs(cost - base_cost))
        avg = sum(deltas) / len(deltas) if deltas else 1.0
        return max(1.0, 3.0 * avg)

    def _perturb(self, state: SearchState, phase: str) -> None:
        choices = ["swap", "move", "rotate", "reshape_mul", "reshape_div"]
        if phase == "high" and state.groups:
            choices.append("subtree")
        move = random.choice(choices)

        if move == "swap":
            state.tree.move_swap_payloads()
        elif move == "move":
            state.tree.move_delete_insert()
        elif move == "subtree":
            random.choice(list(state.groups.values())).tree.move_delete_insert()
        else:
            keys = state.shape.soft_shape_keys()
            if not keys:
                return
            key = random.choice(keys)
            if move == "rotate":
                state.shape.rotate_key(key)
            elif move == "reshape_mul":
                state.shape.reshape_key(key, random.uniform(0.02, 0.12), divide=False)
            else:
                state.shape.reshape_key(key, random.uniform(0.02, 0.12), divide=True)

    def _pack_state(
        self,
        state: SearchState,
        preplaced: Dict[int, Tuple[float, float, float, float]],
        phase: str,
    ) -> List[Tuple[float, float, float, float]]:
        obstacles = list(preplaced.values())
        positions: Dict[int, Tuple[float, float, float, float]] = dict(preplaced)
        group_local: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}

        def item_dims(payload: int) -> Tuple[float, float]:
            if payload >= 0:
                return state.shape.dimensions(payload)
            group = state.groups[-payload]
            local = self._pack_group(group, state.shape)
            group_local[-payload] = local
            max_x = max((x + w for x, y, w, h in local.values()), default=0.0)
            max_y = max((y + h for x, y, w, h in local.values()), default=0.0)
            return max(max_x, EPS), max(max_y, EPS)

        packed_items = state.tree.pack(item_dims, obstacles)
        for payload, packed in packed_items.items():
            if payload >= 0:
                positions[payload] = (packed.x, packed.y, packed.w, packed.h)
            else:
                gid = -payload
                local = group_local.get(gid) or self._pack_group(state.groups[gid], state.shape)
                for bid, (lx, ly, lw, lh) in local.items():
                    positions[bid] = (packed.x + lx, packed.y + ly, lw, lh)

        block_count = len(state.shape.blocks)
        return [positions.get(i, (0.0, 0.0, *state.shape.dimensions(i))) for i in range(block_count)]

    def _pack_group(self, group: GroupMacro, shape: ShapeState) -> Dict[int, Tuple[float, float, float, float]]:
        packed = group.tree.pack(lambda bid: shape.dimensions(bid), obstacles=None)
        return {bid: (p.x, p.y, p.w, p.h) for bid, p in packed.items()}

    def _cost(
        self,
        positions: List[Tuple[float, float, float, float]],
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        blocks: Dict[int, BlockInfo],
        phase: str,
    ) -> float:
        if not self._hard_feasible(positions, blocks):
            return INF

        total_area = sum(max(block.area, EPS) for block in blocks.values())
        hpwl = calculate_hpwl_b2b(positions, b2b) + calculate_hpwl_p2b(positions, p2b, pins)
        net_weight = self._net_weight(b2b) + self._net_weight(p2b)
        hpwl_norm = hpwl / max(math.sqrt(total_area) * max(net_weight, 1.0), EPS)
        area_norm = calculate_bbox_area(positions) / max(total_area, EPS)
        boundary_norm = self._boundary_penalty(positions, blocks) / max(math.sqrt(total_area), EPS)

        if phase == "high":
            return 1.20 * hpwl_norm + 1.00 * area_norm + 0.60 * boundary_norm

        group_norm = self._group_penalty(positions, blocks) / max(total_area, EPS)
        official_soft = self._official_soft_violation_proxy(positions, blocks)
        return (
            1.00 * hpwl_norm
            + 1.15 * area_norm
            + 0.50 * boundary_norm
            + 0.35 * group_norm
            + 0.25 * official_soft
        )

    def _hard_feasible(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Dict[int, BlockInfo],
    ) -> bool:
        # The contour packer provides overlap-free placement by construction.
        # Avoid O(n^2) pair checks inside every SA proposal; the official
        # evaluator still performs the exact overlap check on the final output.
        for bid, block in blocks.items():
            x, y, w, h = positions[bid]
            if block.is_preplaced:
                if (
                    abs(x - block.target_x) > 1e-4
                    or abs(y - block.target_y) > 1e-4
                    or abs(w - block.target_w) > 1e-4
                    or abs(h - block.target_h) > 1e-4
                ):
                    return False
            elif block.is_fixed:
                if abs(w - block.target_w) > 1e-4 or abs(h - block.target_h) > 1e-4:
                    return False
            else:
                if abs(w * h - block.area) / max(block.area, EPS) > 0.011:
                    return False
        return True

    def _boundary_penalty(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Dict[int, BlockInfo],
    ) -> float:
        x_min = min(x for x, y, w, h in positions)
        y_min = min(y for x, y, w, h in positions)
        x_max = max(x + w for x, y, w, h in positions)
        y_max = max(y + h for x, y, w, h in positions)
        penalty = 0.0
        for bid, block in blocks.items():
            code = block.boundary
            if code == 0:
                continue
            x, y, w, h = positions[bid]
            if code & 1:
                penalty += abs(x - x_min)
            if code & 2:
                penalty += abs(x + w - x_max)
            if code & 4:
                penalty += abs(y + h - y_max)
            if code & 8:
                penalty += abs(y - y_min)
        return penalty

    def _group_penalty(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Dict[int, BlockInfo],
    ) -> float:
        by_group: Dict[int, List[int]] = {}
        for bid, block in blocks.items():
            if block.cluster > 0:
                by_group.setdefault(block.cluster, []).append(bid)

        penalty = 0.0
        for members in by_group.values():
            if len(members) < 2:
                continue
            x_min = min(positions[i][0] for i in members)
            y_min = min(positions[i][1] for i in members)
            x_max = max(positions[i][0] + positions[i][2] for i in members)
            y_max = max(positions[i][1] + positions[i][3] for i in members)
            bbox_area = (x_max - x_min) * (y_max - y_min)
            block_area = sum(positions[i][2] * positions[i][3] for i in members)
            penalty += max(0.0, bbox_area - block_area)
        return penalty

    def _official_soft_violation_proxy(
        self,
        positions: List[Tuple[float, float, float, float]],
        blocks: Dict[int, BlockInfo],
    ) -> float:
        n_soft = 0
        violation = 0

        for bid, block in blocks.items():
            if block.boundary:
                n_soft += 1
        boundary_dist = self._boundary_penalty(positions, blocks)
        if boundary_dist > 1e-5:
            violation += 1

        for groups in (self._ids_by_attr(blocks, "cluster"), self._ids_by_attr(blocks, "mib")):
            for members in groups.values():
                if len(members) > 1:
                    n_soft += len(members) - 1

        for members in self._ids_by_attr(blocks, "mib").values():
            shapes = {(round(positions[i][2], 4), round(positions[i][3], 4)) for i in members}
            violation += max(0, len(shapes) - 1)

        for members in self._ids_by_attr(blocks, "cluster").values():
            violation += max(0, self._edge_contact_components(positions, members) - 1)

        return violation / max(1, n_soft)

    def _ids_by_attr(self, blocks: Dict[int, BlockInfo], attr: str) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for bid, block in blocks.items():
            key = getattr(block, attr)
            if key > 0:
                out.setdefault(key, []).append(bid)
        return out

    def _edge_contact_components(
        self,
        positions: List[Tuple[float, float, float, float]],
        members: List[int],
    ) -> int:
        if not members:
            return 0
        parent = {i: i for i in members}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for idx, a in enumerate(members):
            ax, ay, aw, ah = positions[a]
            for b in members[idx + 1:]:
                bx, by, bw, bh = positions[b]
                y_overlap = min(ay + ah, by + bh) - max(ay, by)
                x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
                vertical_touch = abs(ax + aw - bx) < 1e-5 or abs(bx + bw - ax) < 1e-5
                horizontal_touch = abs(ay + ah - by) < 1e-5 or abs(by + bh - ay) < 1e-5
                if (vertical_touch and y_overlap > 1e-5) or (horizontal_touch and x_overlap > 1e-5):
                    union(a, b)
        return len({find(i) for i in members})

    def _net_weight(self, conn: torch.Tensor) -> float:
        if conn is None or len(conn) == 0:
            return 0.0
        total = 0.0
        for edge in conn:
            if edge[0] == -1:
                continue
            total += float(edge[2]) if len(edge) > 2 else 1.0
        return total

    def _fallback_shelf(
        self,
        block_count: int,
        blocks: Dict[int, BlockInfo],
    ) -> List[Tuple[float, float, float, float]]:
        positions: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 1.0, 1.0)] * block_count
        x_cursor = 0.0
        max_h = 0.0
        for bid in range(block_count):
            block = blocks[bid]
            if block.is_preplaced:
                positions[bid] = (block.target_x, block.target_y, block.target_w, block.target_h)
                continue
            if block.is_fixed:
                w, h = block.target_w, block.target_h
            else:
                w = h = math.sqrt(max(block.area, EPS))
            while any(rects_overlap(x_cursor, 0.0, w, h, *positions[j]) for j in range(bid) if j < len(positions)):
                x_cursor += w
            positions[bid] = (x_cursor, 0.0, w, h)
            x_cursor += w
            max_h = max(max_h, h)
        return positions
