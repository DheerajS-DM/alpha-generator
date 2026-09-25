"""
Unit tests for AlphaEvolver.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_evolver import AlphaEvolver
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast
from local_alpha_engine import compile_to_polars
from ast_node import parse_formula

class TestAlphaEvolver(unittest.TestCase):
    def setUp(self):
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "operatorRAW.json")
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_processed.csv")
        self.registry = OperatorRegistry(json_path)
        self.valid_fields = ["open", "high", "low", "close", "volume", "vwap"]
        self.evolver = AlphaEvolver(
            self.registry,
            valid_fields=self.valid_fields,
            population_size=30,
            max_depth=6
        )
        self.csv_path = csv_path

    def test_seeding_from_logs(self):
        if not os.path.exists(self.csv_path):
            self.skipTest("logs_processed.csv not present on disk")
        loaded = self.evolver.seed_from_logs(self.csv_path, top_n=10)
        self.assertGreater(loaded, 0)
        self.assertGreaterEqual(len(self.evolver.population), loaded)

    def test_seeding_from_templates(self):
        self.evolver.seed_from_templates(count=15)
        self.assertGreaterEqual(len(self.evolver.population), 15)

    def test_crossover(self):
        p1 = parse_formula("ts_decay_linear(rank(ts_zscore(close, 20)), 10)")
        p2 = parse_formula("hump(zscore(subtract(high, low)))")
        c1, c2 = self.evolver.crossover(p1, p2)
        
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertLessEqual(c1.depth(), self.evolver.max_depth)
        self.assertLessEqual(c2.depth(), self.evolver.max_depth)

    def test_mutate(self):
        ast = parse_formula("ts_decay_linear(rank(ts_zscore(close, 20)), 10)")
        mutated = self.evolver.mutate(ast)
        self.assertIsNotNone(mutated)
        self.assertLessEqual(mutated.depth(), self.evolver.max_depth)

    def test_generate_batch(self):
        self.evolver.seed_from_logs(self.csv_path, top_n=10)
        batch = self.evolver.generate_batch(count=5)
        self.assertEqual(len(batch), 5)
        for cand in batch:
            is_valid, errors = validate_ast(cand, self.registry, self.valid_fields)
            self.assertTrue(is_valid, f"Evolved candidate is invalid: {errors} in {cand.to_string()}")
            expr = compile_to_polars(cand)
            self.assertIsNotNone(expr)

if __name__ == '__main__':
    unittest.main()
