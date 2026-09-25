"""
Unit tests for generate_candidate_batch across all strategies.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_background_runner import generate_candidate_batch
from local_alpha_engine import AlphaGenerator
from alpha_templates import AlphaTemplateGenerator
from alpha_evolver import AlphaEvolver
from alpha_healer import AlphaHealer
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast

class TestCandidateBatch(unittest.TestCase):
    def setUp(self):
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "operatorRAW.json")
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_processed.csv")
        self.registry = OperatorRegistry(json_path)
        self.valid_fields = ["open", "high", "low", "close", "volume"]
        self.random_gen = AlphaGenerator(self.registry, valid_fields=self.valid_fields)
        self.template_gen = AlphaTemplateGenerator(valid_fields=self.valid_fields)
        self.evolver = AlphaEvolver(self.registry, valid_fields=self.valid_fields, population_size=30)
        self.evolver.seed_from_logs(csv_path, top_n=10)
        self.evolver.seed_from_templates(count=10)
        self.healer = AlphaHealer(self.registry, logs_csv_path=csv_path, valid_fields=self.valid_fields)

    def test_all_strategies_batch_generation(self):
        strategies = ["hybrid", "templates", "evolve", "heal", "random"]
        batch_size = 5
        for strat in strategies:
            batch = generate_candidate_batch(
                strategy=strat,
                template_gen=self.template_gen,
                evolver=self.evolver,
                healer=self.healer,
                random_gen=self.random_gen,
                registry=self.registry,
                valid_fields=self.valid_fields,
                batch_size=batch_size
            )
            self.assertEqual(len(batch), batch_size, f"Strategy '{strat}' returned {len(batch)} items, expected {batch_size}")
            for ast_node, formula_str, source_tag in batch:
                is_valid, errors = validate_ast(ast_node, self.registry, self.valid_fields)
                self.assertTrue(is_valid, f"Strategy '{strat}' produced invalid AST: {errors} in {formula_str}")
                self.assertIsNotNone(source_tag)

if __name__ == '__main__':
    unittest.main()
