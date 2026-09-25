"""
Unit tests for AlphaTemplateGenerator.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_templates import AlphaTemplateGenerator
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast
from local_alpha_engine import compile_to_polars

class TestAlphaTemplates(unittest.TestCase):
    def setUp(self):
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "operatorRAW.json")
        self.registry = OperatorRegistry(json_path)
        self.valid_fields = ["open", "high", "low", "close", "volume"]
        self.generator = AlphaTemplateGenerator(self.valid_fields)

    def test_all_scenarios_generate_and_compile(self):
        scenarios = self.generator.get_available_scenarios()
        for sc in scenarios:
            for _ in range(5):
                ast = self.generator.generate(scenario=sc, apply_wrappers=True)
                self.assertIsNotNone(ast)
                
                # Check validation
                valid, errors = validate_ast(ast, self.registry, self.valid_fields)
                self.assertTrue(valid, f"Validation failed for scenario '{sc}': {errors} in formula {ast.to_string()}")

                # Check compilation to polars expression
                try:
                    expr = compile_to_polars(ast)
                    self.assertIsNotNone(expr)
                except Exception as e:
                    self.fail(f"Compilation to Polars failed for scenario '{sc}': {e} in formula {ast.to_string()}")

    def test_random_scenario_generation(self):
        for _ in range(20):
            ast = self.generator.generate(scenario=None, apply_wrappers=True)
            valid, errors = validate_ast(ast, self.registry, self.valid_fields)
            self.assertTrue(valid, f"Validation failed: {errors} in {ast.to_string()}")

if __name__ == '__main__':
    unittest.main()
