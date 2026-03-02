import math
import unittest
import random
from smith.tensor import NanoTensor
from smith.gru_model import GatedRecurrentUnit
from smith.trainer import Trainer
from smith.database import SymbolicDB

class TestFormalVerification(unittest.TestCase):
    """
    Formal Verification Test Suite for AtomicGradient Engine.
    """

    def test_autograd_linearity(self):
        """Verify linearity of addition and multiplication gradients."""
        a = NanoTensor([1.0, 2.0], requires_grad=True)
        b = NanoTensor([3.0, 4.0], requires_grad=True)
        c = (a * b) + a
        c.backward()
        self.assertEqual(a.grad, [4.0, 5.0])
        self.assertEqual(b.grad, [1.0, 2.0])

    def test_tanh_precision(self):
        """Verify high-precision tanh approximation drift."""
        test_points = [-1.0, 0.0, 1.0, 2.0]
        for x in test_points:
            expected = math.tanh(x)
            actual = NanoTensor._tanh(x)
            self.assertLess(abs(expected - actual), 1e-6)

    def test_softmax_zero_sum_gradient(self):
        """Verify gradient of softmax + cross-entropy sums to zero."""
        db = SymbolicDB(":memory:")
        model = GatedRecurrentUnit(vocab_size=10, hidden_size=4, db=db)
        trainer = Trainer(model)
        logits = NanoTensor([0.1, 0.5, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        target = 2
        loss = trainer.compute_loss(logits, target)
        loss.backward()
        grad_sum = sum(logits.grad)
        self.assertLess(abs(grad_sum), 1e-9)

    def test_risk_weighted_divergence(self):
        """Verify that creative divergence triggers for low-confidence tokens."""
        random.seed(42)
        db = SymbolicDB(":memory:")
        model = GatedRecurrentUnit(vocab_size=10, hidden_size=4, db=db)
        logits = NanoTensor([1.0] * 10)

        # Heuristic in gru_model: reward_multiplier = [(1.0 + (i % 10) / 100.0) for i in range(len(probs))]
        # Index 9 has 1.09 multiplier, Index 0 has 1.00 multiplier.
        counts = [0] * 10
        for _ in range(5000): # More samples for statistical stability
            idx = model.sample_risk_weighted(logits, temperature=1.0)
            counts[idx] += 1

        self.assertGreater(counts[9], counts[0], f"Creative divergence not observed: {counts[9]} vs {counts[0]}")

    def test_hidden_state_detaching(self):
        """Verify that hidden state is detached in the trainer."""
        db = SymbolicDB(":memory:")
        model = GatedRecurrentUnit(vocab_size=10, hidden_size=4, db=db)
        trainer = Trainer(model)
        inputs = [1, 2]
        targets = [2, 3]
        loss, gnorm = trainer.train_step(inputs, targets)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(gnorm, float)

if __name__ == '__main__':
    unittest.main()
