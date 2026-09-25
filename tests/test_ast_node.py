"""
Unit tests for ASTNode methods and parse_formula.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ast_node import ASTNode, parse_formula

class TestASTNode(unittest.TestCase):
    def test_basic_ast_and_to_string(self):
        leaf1 = ASTNode(type='field', value='close')
        leaf2 = ASTNode(type='constant', value='10')
        op = ASTNode(type='operator', value='ts_delta', children=[leaf1, leaf2])
        self.assertEqual(op.to_string(), "ts_delta(close, 10)")
        self.assertEqual(op.depth(), 2)
        self.assertEqual(op.size(), 3)

    def test_clone_deep_copy(self):
        root = ASTNode(
            type='operator',
            value='rank',
            children=[
                ASTNode(type='operator', value='ts_mean', children=[
                    ASTNode(type='field', value='volume'),
                    ASTNode(type='constant', value='20')
                ])
            ]
        )
        cloned = root.clone()
        self.assertEqual(cloned.to_string(), root.to_string())
        # Verify modifying clone does not modify original
        cloned.children[0].children[0].value = 'open'
        self.assertEqual(root.children[0].children[0].value, 'volume')
        self.assertEqual(cloned.children[0].children[0].value, 'open')

    def test_collect_subtrees_with_context(self):
        formula = "ts_decay_linear(rank(ts_zscore(close, 20)), 10)"
        node = parse_formula(formula)
        subtrees = node.collect_subtrees_with_context()
        self.assertGreater(len(subtrees), 4)
        root_parent, root_idx, root_node = subtrees[0]
        self.assertIsNone(root_parent)
        self.assertEqual(root_idx, -1)
        self.assertEqual(root_node.value, "ts_decay_linear")

    def test_parse_formula_roundtrip(self):
        cases = [
            "close",
            "10",
            "-0.5",
            "rank(close)",
            "ts_delta(volume, 5)",
            "ts_corr(close, volume, 20)",
            "ts_decay_linear(rank(ts_zscore(close, 20)), 10)",
            "add(ts_delta(high, 5), subtract(low, close))",
            "hump(rank(ts_mean(ts_std_dev(reverse(ts_sum(multiply(volume, high), 34)), 52), 5)))"
        ]
        for c in cases:
            ast = parse_formula(c)
            self.assertEqual(c.replace(" ", ""), ast.to_string().replace(" ", ""))

if __name__ == '__main__':
    unittest.main()
