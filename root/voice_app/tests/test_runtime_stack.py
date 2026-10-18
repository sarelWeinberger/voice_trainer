import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RuntimeStackTests(unittest.TestCase):
    def test_runtime_import_stack_is_consistent(self):
        import mediapipe as mp
        import numba
        import numpy

        self.assertTrue(hasattr(mp, "solutions"))
        self.assertTrue(hasattr(mp.solutions, "pose"))
        self.assertLess(tuple(int(part) for part in numpy.__version__.split(".")[:2]), (2, 4))
        self.assertIsNotNone(numba.__version__)

        import app  # noqa: F401


if __name__ == "__main__":
    unittest.main()
