"""Focused tests for boundary/grouping overlays in the visualizer."""

import unittest

import torch

from .visualize import violation_diagnostics


def make_case(cons):
    return {'cons': torch.tensor(cons, dtype=torch.long)}


class VisualizeViolationTest(unittest.TestCase):
    def test_boundary_block_inside_required_side_is_marked(self):
        xywh = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ], dtype=torch.float64)
        case = make_case([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],  # block 1 requires the left bbox side
        ])
        result = violation_diagnostics(xywh, case)
        self.assertEqual(result['boundary_blocks'], [1])
        self.assertEqual(result['boundary_violations'], 1)

    def test_boundary_corner_requires_both_sides(self):
        xywh = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
        ], dtype=torch.float64)
        case = make_case([
            [0, 0, 0, 0, 5],  # left is satisfied, top is not
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(
            violation_diagnostics(xywh, case)['boundary_blocks'], [0])

    def test_shared_edge_connects_group(self):
        xywh = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ], dtype=torch.float64)
        case = make_case([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ])
        result = violation_diagnostics(xywh, case)
        self.assertEqual(result['group_components'], {})
        self.assertEqual(result['grouping_violations'], 0)

    def test_corner_touch_does_not_connect_group(self):
        xywh = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float64)
        case = make_case([
            [0, 0, 0, 2, 0],
            [0, 0, 0, 2, 0],
        ])
        result = violation_diagnostics(xywh, case)
        self.assertEqual(result['group_components'], {2: [[0], [1]]})
        self.assertEqual(result['grouping_violations'], 1)


if __name__ == '__main__':
    unittest.main()
