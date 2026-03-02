import math
import unittest
import os
import shutil
from smith import SymbolicDB, GatedRecurrentUnit, Trainer

class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.text = "abcde" * 10
        self.db_path1 = "test_det1.db"
        self.db_path2 = "test_det2.db"
        for p in [self.db_path1, self.db_path2]:
            if os.path.exists(p):
                os.remove(p)

    def tearDown(self):
        for p in [self.db_path1, self.db_path2]:
            if os.path.exists(p):
                os.remove(p)

    def run_training(self, db_path):
        db = SymbolicDB(db_path)
        model = GatedRecurrentUnit(vocab_size=128, hidden_size=8, db=db)

        # Use small weights to avoid NaNs
        for p in model.params.values():
            p.data = [0.001 * i for i in range(len(p.data))]

        trainer = Trainer(model, learning_rate=0.001, verbose=False)
        trainer.train(self.text, epochs=5, seq_length=5)

        final_params = {k: v.data[:] for k, v in model.params.items()}
        history = trainer.get_history()
        db.close()
        return final_params, history

    def test_bit_for_bit_determinism(self):
        params1, history1 = self.run_training(self.db_path1)
        params2, history2 = self.run_training(self.db_path2)

        for k in params1:
            self.assertEqual(params1[k], params2[k], f"Mismatch in parameter {k}")

        for i in range(len(history1)):
            self.assertEqual(history1[i]['loss'], history2[i]['loss'], f"Loss mismatch at epoch {i+1}")

if __name__ == "__main__":
    unittest.main()
