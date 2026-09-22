import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

# Hardcoded arity for operators we implement in Polars.
# This acts as both the whitelist of what the generator can use,
# and the exact signature (how many expressions, how many constant params) it requires.
# We exclude ts_max and ts_min as they are restricted.
SUPPORTED_OPS = {
    # Arithmetic (14)
    'add': {'exprs': 2, 'consts': 0},
    'subtract': {'exprs': 2, 'consts': 0},
    'multiply': {'exprs': 2, 'consts': 0},
    'divide': {'exprs': 2, 'consts': 0},
    'power': {'exprs': 2, 'consts': 0},
    'signed_power': {'exprs': 2, 'consts': 0},
    'sqrt': {'exprs': 1, 'consts': 0},
    'log': {'exprs': 1, 'consts': 0},
    'inverse': {'exprs': 1, 'consts': 0},
    'min': {'exprs': 2, 'consts': 0},
    'max': {'exprs': 2, 'consts': 0},
    'abs': {'exprs': 1, 'consts': 0},
    'sign': {'exprs': 1, 'consts': 0},
    'reverse': {'exprs': 1, 'consts': 0},

    # Time Series (17)
    'ts_mean': {'exprs': 1, 'consts': 1},
    'ts_delay': {'exprs': 1, 'consts': 1},
    'ts_delta': {'exprs': 1, 'consts': 1},
    'ts_std_dev': {'exprs': 1, 'consts': 1},
    'ts_sum': {'exprs': 1, 'consts': 1},
    'ts_decay_linear': {'exprs': 1, 'consts': 1},
    'ts_corr': {'exprs': 2, 'consts': 1},
    'ts_covariance': {'exprs': 2, 'consts': 1},
    'ts_scale': {'exprs': 1, 'consts': 1},
    'ts_zscore': {'exprs': 1, 'consts': 1},
    'ts_product': {'exprs': 1, 'consts': 1},
    'ts_av_diff': {'exprs': 1, 'consts': 1},
    'ts_count_nans': {'exprs': 1, 'consts': 1},
    'hump': {'exprs': 1, 'consts': 0},  # hump takes exactly 1 input in WQ Brain

    # Cross Sectional (6)
    'rank': {'exprs': 1, 'consts': 0},
    'normalize': {'exprs': 1, 'consts': 0},
    'zscore': {'exprs': 1, 'consts': 0},
    'winsorize': {'exprs': 1, 'consts': 0},
    'scale': {'exprs': 1, 'consts': 0},
    'quantile': {'exprs': 1, 'consts': 0},
}

@dataclass
class OperatorMeta:
    name: str
    category: str
    scope: List[str]
    level: Optional[str]
    definition: str
    description: str
    expr_args: int
    const_args: int
    has_polars_impl: bool

class OperatorRegistry:
    def __init__(self, json_path: str):
        self.operators: Dict[str, OperatorMeta] = {}
        self._load_from_json(json_path)
        
    def _load_from_json(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_ops = json.load(f)
            
        for raw in raw_ops:
            name = raw.get('name', '')
            if not name:
                continue
                
            has_impl = name in SUPPORTED_OPS
            expr_args = SUPPORTED_OPS[name]['exprs'] if has_impl else 1
            const_args = SUPPORTED_OPS[name]['consts'] if has_impl else 0
            
            meta = OperatorMeta(
                name=name,
                category=raw.get('category', 'Unknown'),
                scope=raw.get('scope', []),
                level=raw.get('level'),
                definition=raw.get('definition', ''),
                description=raw.get('description', ''),
                expr_args=expr_args,
                const_args=const_args,
                has_polars_impl=has_impl
            )
            self.operators[name] = meta

    def get_all(self) -> List[OperatorMeta]:
        return list(self.operators.values())

    def get_beginner_usable(self) -> List[OperatorMeta]:
        """Returns operators available to ALL users with REGULAR scope."""
        return [
            op for op in self.operators.values()
            if op.level == "ALL" and "REGULAR" in op.scope
        ]

    def get_implementable(self) -> List[OperatorMeta]:
        """Returns beginner operators that also have a Polars implementation."""
        return [
            op for op in self.get_beginner_usable()
            if op.has_polars_impl
        ]

    def get_random_operator(self, category_weights: Dict[str, float] = None) -> OperatorMeta:
        """
        Picks a random implementable operator, optionally weighted by category.
        """
        ops = self.get_implementable()
        if not category_weights:
            return random.choice(ops)
            
        # Group ops by category
        by_cat = {}
        for op in ops:
            by_cat.setdefault(op.category, []).append(op)
            
        categories = list(category_weights.keys())
        weights = [category_weights[c] for c in categories]
        
        # Pick category
        chosen_cat = random.choices(categories, weights=weights, k=1)[0]
        
        # If chosen category has no implemented ops, fallback to uniform random
        if chosen_cat not in by_cat or not by_cat[chosen_cat]:
            return random.choice(ops)
            
        return random.choice(by_cat[chosen_cat])
        
    def get_operator(self, name: str) -> Optional[OperatorMeta]:
        return self.operators.get(name)
