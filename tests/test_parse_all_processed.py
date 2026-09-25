"""
Verify parse_formula on all 1146 formulas in logs_processed.csv.
"""
import unittest
import sys
import os
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_node import ASTNode, parse_formula

class TestParseAllProcessed(unittest.TestCase):
    def test_parse_all_logs_processed(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_processed.csv")
        if not os.path.exists(csv_path):
            self.skipTest("logs_processed.csv not found")

        df = pl.read_csv(csv_path)
        formulas = df["formula"].to_list()
        
        parsed_count = 0
        for f in formulas:
            try:
                node = parse_formula(f)
                self.assertIsNotNone(node)
                self.assertEqual(node.to_string().replace(" ", ""), f.replace(" ", ""))
                parsed_count += 1
            except Exception as e:
                self.fail(f"Failed to parse formula '{f}': {e}")

        print(f"\nSuccessfully parsed and round-tripped {parsed_count} / {len(formulas)} formulas from logs_processed.csv")

if __name__ == '__main__':
    unittest.main()
