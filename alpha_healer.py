"""
Alpha Healer: Identifies near-miss alphas from historical logs and systematically
applies targeted quantitative fixes (window tuning, neutralization swap, sign reversal, smoothing).
"""
import os
import random
import polars as pl
from typing import List, Optional, Tuple, Dict, Any
from ast_node import ASTNode, parse_formula
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast

class AlphaHealer:
    """
    Finds promising near-miss alphas in logs_processed.csv and produces healed variants.
    """

    def __init__(
        self,
        registry: OperatorRegistry,
        logs_csv_path: str = "logs_processed.csv",
        valid_fields: Optional[List[str]] = None
    ):
        self.registry = registry
        self.logs_csv_path = logs_csv_path
        self.valid_fields = valid_fields or ["open", "high", "low", "close", "volume"]
        self.near_misses: List[Dict[str, Any]] = []
        self._load_near_misses()

    def _load_near_misses(self):
        """Loads and filters promising near-miss candidates from the processed logs CSV."""
        if not os.path.exists(self.logs_csv_path):
            return

        try:
            df = pl.read_csv(self.logs_csv_path)
            if len(df) == 0 or 'formula' not in df.columns or 'median_sharpe' not in df.columns:
                return

            # Clean and filter
            for row in df.iter_rows(named=True):
                formula = row.get('formula')
                if not formula or not isinstance(formula, str):
                    continue

                sharpe = float(row.get('median_sharpe') or 0.0)
                worst_sharpe = float(row.get('worst_sharpe') or -99.0)
                consistency = float(row.get('consistency') or 0.0)

                # Criteria for near-miss:
                # 1. Close to Sharpe threshold (e.g. 0.45 <= sharpe < 1.0)
                # 2. Inverted predictor: strong negative Sharpe (sharpe <= -0.5)
                # 3. High consistency but slightly low Sharpe (consistency >= 60% and sharpe >= 0.3)
                is_near_miss = (
                    (0.45 <= sharpe < 1.0 and worst_sharpe > -1.5) or
                    (sharpe <= -0.5 and worst_sharpe < -1.0) or
                    (consistency >= 60.0 and sharpe >= 0.3)
                )

                if is_near_miss:
                    self.near_misses.append({
                        'formula': formula,
                        'median_sharpe': sharpe,
                        'worst_sharpe': worst_sharpe,
                        'consistency': consistency
                    })

        except Exception as e:
            # If reading CSV fails, keep candidate pool empty
            pass

    def get_candidate_count(self) -> int:
        return len(self.near_misses)

    def heal_formula(self, formula: str, sharpe_hint: float = 0.5) -> List[ASTNode]:
        """
        Takes a formula string and produces a list of healed ASTNode variants.
        """
        try:
            root = parse_formula(formula)
        except Exception:
            return []

        healed = []

        # 1. Sign reversal if inverted or negative
        if sharpe_hint < 0 or random.random() < 0.25:
            reversed_ast = self._heal_sign_reversal(root)
            if reversed_ast:
                healed.append(reversed_ast)

        # 2. Window scaling / tuning
        scaled_ast = self._heal_adjust_windows(root)
        if scaled_ast:
            healed.append(scaled_ast)

        # 3. Neutralization swap / wrapping
        neut_ast = self._heal_neutralization(root)
        if neut_ast:
            healed.append(neut_ast)

        # 4. Smoothing injection / dampening
        smooth_ast = self._heal_smoothing(root)
        if smooth_ast:
            healed.append(smooth_ast)

        # 5. Operator upgrades (e.g., ts_mean -> ts_decay_linear)
        upgrade_ast = self._heal_operator_upgrade(root)
        if upgrade_ast:
            healed.append(upgrade_ast)

        # Filter valid ASTs
        valid_healed = []
        seen_strings = {root.to_string()}
        for ast in healed:
            s = ast.to_string()
            if s not in seen_strings:
                is_valid, _ = validate_ast(ast, self.registry, self.valid_fields)
                if is_valid:
                    seen_strings.add(s)
                    valid_healed.append(ast)

        return valid_healed

    def _heal_sign_reversal(self, root: ASTNode) -> ASTNode:
        """Inverts the formula: if root is reverse(x), unwraps it; otherwise wraps in reverse()."""
        cloned = root.clone()
        if cloned.type == 'operator' and cloned.value == 'reverse' and cloned.children:
            return cloned.children[0]
        return ASTNode(type='operator', value='reverse', children=[cloned])

    def _heal_adjust_windows(self, root: ASTNode) -> ASTNode:
        """Systematically scales window constants by factors (0.5, 1.5, 2.0)."""
        cloned = root.clone()
        scale_factor = random.choice([0.5, 1.5, 2.0])

        modified = False
        for node in cloned.collect_nodes():
            if node.type == 'operator' and node.children:
                meta = self.registry.get_operator(node.value)
                if meta and meta.const_args > 0:
                    for i in range(meta.expr_args, len(node.children)):
                        c_node = node.children[i]
                        try:
                            val = int(c_node.value)
                            new_val = int(round(val * scale_factor))
                            new_val = max(2, min(252, new_val))
                            if new_val != val:
                                c_node.value = str(new_val)
                                modified = True
                        except ValueError:
                            pass
        return cloned if modified else None

    def _heal_neutralization(self, root: ASTNode) -> ASTNode:
        """Swaps or injects cross-sectional ranking operators (rank, zscore, normalize)."""
        cloned = root.clone()
        cross_ops = ['rank', 'zscore', 'normalize']

        # If root is already a cross-sectional op, swap it
        if cloned.type == 'operator' and cloned.value in cross_ops:
            alt_ops = [op for op in cross_ops if op != cloned.value]
            cloned.value = random.choice(alt_ops)
            return cloned

        # Otherwise wrap root in rank
        return ASTNode(type='operator', value='rank', children=[cloned])

    def _heal_smoothing(self, root: ASTNode) -> ASTNode:
        """Applies turnover dampening: adds ts_decay_linear or hump to curb excess volatility."""
        cloned = root.clone()

        # If already wrapped in smoothing, adjust decay window
        if cloned.type == 'operator' and cloned.value == 'ts_decay_linear' and len(cloned.children) >= 2:
            current_w = int(cloned.children[1].value)
            new_w = max(2, min(30, current_w + random.choice([-2, 2, 5])))
            cloned.children[1].value = str(new_w)
            return cloned

        # Otherwise wrap in ts_decay_linear or hump, then apply outer neutralization
        if random.random() < 0.6:
            decay_window = random.choice([3, 5, 8, 10])
            param_node = ASTNode(type='constant', value=str(decay_window))
            smoothed = ASTNode(type='operator', value='ts_decay_linear', children=[cloned, param_node])
        else:
            smoothed = ASTNode(type='operator', value='hump', children=[cloned])
            
        return ASTNode(type='operator', value=random.choice(['rank', 'zscore', 'normalize']), children=[smoothed])

    def _heal_operator_upgrade(self, root: ASTNode) -> ASTNode:
        """Upgrades weaker operators to stronger ones: e.g. ts_mean -> ts_decay_linear."""
        cloned = root.clone()
        modified = False

        for node in cloned.collect_nodes():
            if node.type == 'operator' and node.value == 'ts_mean' and len(node.children) == 2:
                # Replace simple moving average with linearly decaying moving average
                node.value = 'ts_decay_linear'
                modified = True
                break
            elif node.type == 'operator' and node.value == 'ts_delta' and len(node.children) == 2:
                # Replace raw delta with standardized zscore
                node.value = 'ts_zscore'
                modified = True
                break

        return cloned if modified else None

    def generate_healed_batch(self, batch_size: int = 5) -> List[ASTNode]:
        """
        Generates a batch of healed ASTNode candidates from the near-miss pool.
        """
        if not self.near_misses:
            return []

        results = []
        attempts = 0
        max_attempts = batch_size * 5

        while len(results) < batch_size and attempts < max_attempts:
            attempts += 1
            item = random.choice(self.near_misses)
            variants = self.heal_formula(item['formula'], sharpe_hint=item.get('median_sharpe', 0.5))
            if variants:
                chosen = random.choice(variants)
                results.append(chosen)

        return results
