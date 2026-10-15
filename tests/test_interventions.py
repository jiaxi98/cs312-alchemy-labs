"""Check the location and exact size of the P4 training-data perturbation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from interventions import edit_tokens


class Dataset:
    def __init__(self):
        self.tokens = np.zeros((40, 128), dtype=np.uint16)
        self.tokens[20, 101] = 17
        self.seq_len = 128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, row):
        return {"input_ids": self.tokens[row]}


class InterventionTest(unittest.TestCase):
    def test_later_edit_changes_only_declared_tokens_after_prior_batches(self):
        dataset = Dataset()
        before = dataset.tokens.copy()
        audit = edit_tokens(dataset, after_steps=5, count=4, position=100, batch_size=4, vocab_size=4096)
        locations = np.argwhere(before != dataset.tokens)
        np.testing.assert_array_equal(locations, [[20, 100], [20, 101], [20, 102], [20, 103]])
        np.testing.assert_array_equal(dataset.tokens[:20], before[:20])
        self.assertEqual(audit["first_consumed_step"], 6)
        self.assertEqual(audit["after"], [17, 18, 17, 17])

    def test_invalid_edit_cannot_silently_miss_training(self):
        with self.assertRaises(ValueError):
            edit_tokens(Dataset(), after_steps=10, count=1, position=100, batch_size=4, vocab_size=4096)
        with self.assertRaises(ValueError):
            edit_tokens(Dataset(), after_steps=0, count=40, position=100, batch_size=4, vocab_size=4096)


if __name__ == "__main__":
    unittest.main()
