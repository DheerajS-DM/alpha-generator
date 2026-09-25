"""
Unit tests for AlphaHealer.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_healer import AlphaHealer
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast
from local_alpha_engine import compile_to_polars

class TestAlphaHealer(unittest.TestCase):
    def setUp(self):
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "operatorRAW.json")
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_processed.csv")
        self.registry = OperatorRegistry(json_path)
        self.valid_fields = ["open", "high", "low", "close", "volume", "vwap"]
        self.healer = AlphaHealer(self.registry, logs_csv_path=csv_path, valid_fields=self.valid_fields)

    def test_candidates_loaded(self):
        if not os.path.exists(self.healer.logs_csv_path):
            self.skipTest("logs_processed.csv not present on disk")
        count = self.healer.get_candidate_count()
        self.assertGreater(count, 0, "Healer should find near-miss candidates in logs_processed.csv")

    def test_heal_formula_variants(self):
        test_formula = "rank(ts_mean(close, 20))"
        healed_list = self.healer.heal_formula(test_formula, sharpe_hint=0.8)
        self.assertGreater(len(healed_list), 0)

        for ast in healed_list:
            is_valid, errors = validate_ast(ast, self.registry, self.valid_fields)
            self.assertTrue(is_valid, f"Healed AST is invalid: {errors} in {ast.to_string()}")
            expr = compile_to_polars(ast)
            self.assertIsNotNone(expr)

    def test_heal_inverted_formula(self):
        negative_formula = "rank(ts_delta(close, 10))"
        healed_list = self.healer.heal_formula(negative_formula, sharpe_hint=-1.2)
        strings = [h.to_string() for h in healed_list]
        self.assertTrue(any("reverse" in s for s in strings), "Negative formula should produce a reversed variant")

    def test_generate_healed_batch(self):
        if not os.path.exists(self.healer.logs_csv_path):
            self.skipTest("logs_processed.csv not present on disk")
        batch = self.healer.generate_healed_batch(batch_size=5)
        self.assertGreater(len(batch), 0)
        for ast in batch:
            is_valid, errors = validate_ast(ast, self.registry, self.valid_fields)
            self.assertTrue(is_valid, f"Batch AST is invalid: {errors} in {ast.to_string()}")

if __name__ == '__main__':
    unittest.main()
