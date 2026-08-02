import re
import logging
from typing import Dict, List, Optional
import ast

logger = logging.getLogger(__name__)

class RuleBasedTranslator:
    """
    Rule-based translator for WorldQuant Brain formulas to Polars.
    Handles common operators using pattern matching and direct mapping.
    Falls back to LLM for complex/unsupported operators.
    """
    
    def __init__(self):
        # Define operator patterns and their Polars equivalents
        self.operator_patterns = {
            # Arithmetic operators
            r'add\(([^,]+),\s*([^,]+)\)': r'(\1) + (\2)',
            r'subtract\(([^,]+),\s*([^,]+)\)': r'(\1) - (\2)',
            r'multiply\(([^,]+),\s*([^,]+)\)': r'(\1) * (\2)',
            r'divide\(([^,]+),\s*([^,]+)\)': r'(\1) / (\2)',
            r'power\(([^,]+),\s*([^,]+)\)': r'(\1).pow(\2)',
            r'abs\(([^,]+)\)': r'(\1).abs()',
            r'sign\(([^,]+)\)': r'(\1).sign()',
            r'sqrt\(([^,]+)\)': r'(\1).sqrt()',
            r'log\(([^,]+)\)': r'(\1).log()',
            r'reverse\(([^,]+)\)': r'-(\1)',
            
            # Cross-sectional operators (simplified to avoid infinite loops)
            r'normalize\(([^,]+)\)': r'(\1) - (\1).mean()',
            r'zscore\(([^,]+)\)': r'((\1) - (\1).mean()) / (\1).std()',
            
            # Time-series operators (unary with window)
            r'ts_mean\(([^,]+),\s*(\d+)\)': r'(\1).rolling_mean(window_size=\2)',
            r'ts_delay\(([^,]+),\s*(\d+)\)': r'(\1).shift(\2)',
            r'ts_delta\(([^,]+),\s*(\d+)\)': r'(\1) - (\1).shift(\2)',
            r'ts_max\(([^,]+),\s*(\d+)\)': r'(\1).rolling_max(window_size=\2)',
            r'ts_min\(([^,]+),\s*(\d+)\)': r'(\1).rolling_min(window_size=\2)',
            r'ts_std_dev\(([^,]+),\s*(\d+)\)': r'(\1).rolling_std(window_size=\2)',
            r'ts_decay_linear\(([^,]+),\s*(\d+)\)': r'(\1).rolling_mean(window_size=\2)',  # Approximation
            
            # Time-series operators (binary with window)
            r'ts_corr\(([^,]+),\s*([^,]+),\s*(\d+)\)': r'pl.rolling_corr(\1, \2, window_size=\3)',
            r'ts_covariance\(([^,]+),\s*([^,]+),\s*(\d+)\)': r'pl.rolling_cov(\1, \2, window_size=\3)',
        }
        
        # Field name mapping
        self.field_mapping = {
            'open': 'pl.col("open")',
            'high': 'pl.col("high")',
            'low': 'pl.col("low")',
            'close': 'pl.col("close")',
            'volume': 'pl.col("volume")',
        }
    
    def translate_formula(self, formula: str) -> Optional[str]:
        """
        Translate a single WorldQuant formula to Polars using rule-based patterns.
        Returns None if translation fails (caller should use LLM fallback).
        """
        try:
            # Apply operator patterns iteratively (from innermost to outermost)
            polars_expr = formula
            max_iterations = 10  # Prevent infinite loops
            iterations = 0
            
            while iterations < max_iterations:
                old_expr = polars_expr
                for pattern, replacement in self.operator_patterns.items():
                    polars_expr = re.sub(pattern, replacement, polars_expr)
                
                if polars_expr == old_expr:
                    # No more changes made
                    break
                iterations += 1
            
            # After all operators are translated, replace field names with Polars column references
            for field, polars_col in self.field_mapping.items():
                # Use word boundaries to avoid partial matches
                polars_expr = re.sub(r'\b' + field + r'\b', polars_col, polars_expr)
            
            # Validate the result is syntactically valid Polars
            try:
                ast.parse(polars_expr, mode='eval')
                return polars_expr
            except SyntaxError:
                logger.warning(f"Rule-based translation produced invalid syntax: {polars_expr}")
                return None
                
        except Exception as e:
            logger.warning(f"Rule-based translation failed: {e}")
            return None
    
    def translate_batch(self, formulas: List[str]) -> Dict[str, str]:
        """
        Translate a batch of formulas using rule-based patterns.
        Returns a dict of formula -> polars_code for successfully translated formulas.
        """
        results = {}
        
        for formula in formulas:
            polars_code = self.translate_formula(formula)
            if polars_code:
                results[formula] = polars_code
            else:
                logger.info(f"Rule-based translator could not handle: {formula}")
        
        logger.info(f"Rule-based translator handled {len(results)}/{len(formulas)} formulas")
        return results
    
    def can_handle(self, formula: str) -> bool:
        """
        Check if a formula can likely be handled by rule-based translation.
        """
        # Check if formula contains only known operators
        known_operators = set()
        for pattern in self.operator_patterns.keys():
            # Extract operator name from pattern (before opening paren)
            op_name = pattern.split('(')[0]
            known_operators.add(op_name)
        
        # Check if all operators in formula are known
        for op in known_operators:
            if op in formula:
                return True
        
        return False
