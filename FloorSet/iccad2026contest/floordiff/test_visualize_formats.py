"""Tests for prediction/result JSON normalization used by the visualizer."""

import json
import tempfile
import unittest
from pathlib import Path

from .visualize import load_prediction_cases


class VisualizeFormatTest(unittest.TestCase):
    def load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text(json.dumps(payload))
            return load_prediction_cases(path)

    def test_native_floordiff_cases(self):
        entry = {'n': 2, 'positions': [[0, 0, 1, 1], [1, 0, 1, 1]]}
        self.assertEqual(self.load({'cases': {'2': entry}}), {'2': entry})

    def test_official_evaluator_results(self):
        entry = {
            'test_id': 0,
            'block_count': 2,
            'positions': [[0, 0, 1, 1], [1, 0, 1, 1]],
            'cost': 1.0,
        }
        self.assertEqual(
            self.load({'test_results': [entry]}), {'2': entry})

    def test_official_saved_solutions(self):
        entry = {
            'test_id': 0,
            'block_count': 2,
            'positions': [[0, 0, 1, 1], [1, 0, 1, 1]],
        }
        self.assertEqual(self.load({'solutions': [entry]}), {'2': entry})

    def test_missing_positions_reports_evaluator_error(self):
        with self.assertRaisesRegex(ValueError, 'optimizer failed'):
            self.load({'test_results': [{
                'test_id': 4, 'block_count': 25, 'positions': None,
                'error': 'optimizer failed',
            }]})


if __name__ == '__main__':
    unittest.main()
