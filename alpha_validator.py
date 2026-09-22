from typing import List, Tuple
from ast_node import ASTNode
from operator_registry import OperatorRegistry

def validate_ast(node: ASTNode, registry: OperatorRegistry, valid_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates an AST structure against the OperatorRegistry rules and valid fields.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    _validate_node(node, registry, valid_fields, errors)
    return len(errors) == 0, errors

def _validate_node(node: ASTNode, registry: OperatorRegistry, valid_fields: List[str], errors: List[str]):
    if not node:
        return

    if node.type == 'field':
        if node.value not in valid_fields:
            errors.append(f"Invalid field: '{node.value}' not in {valid_fields}")
    
    elif node.type == 'constant':
        try:
            val = float(node.value)
            # In WQ Brain, window parameters are typically >= 2
            # and other parameters (like scale or hump) can be fractions.
            # We just ensure it's a valid number.
        except ValueError:
            errors.append(f"Invalid constant: '{node.value}' is not a number")
            
    elif node.type == 'operator':
        meta = registry.get_operator(node.value)
        if not meta:
            errors.append(f"Unknown operator: '{node.value}'")
            return
            
        if not meta.has_polars_impl:
            errors.append(f"Operator '{node.value}' has no Polars implementation")
            
        if meta.level != "ALL" or "REGULAR" not in meta.scope:
            errors.append(f"Operator '{node.value}' is restricted (not beginner-usable)")

        expected_children_count = meta.expr_args + meta.const_args
        actual_children_count = len(node.children) if node.children else 0
        
        if actual_children_count != expected_children_count:
            errors.append(
                f"Operator '{node.value}' expects {expected_children_count} arguments "
                f"({meta.expr_args} exprs, {meta.const_args} consts), "
                f"got {actual_children_count}"
            )
        else:
            # Check argument types
            for i in range(meta.expr_args):
                child = node.children[i]
                if child.type not in ('operator', 'field', 'constant'): # sometimes expressions can just be a constant, though we generally want fields/operators
                    errors.append(f"Operator '{node.value}' arg {i+1} should be an expression (got {child.type})")
                    
            for i in range(meta.expr_args, expected_children_count):
                child = node.children[i]
                if child.type != 'constant':
                    errors.append(f"Operator '{node.value}' arg {i+1} should be a constant (got {child.type})")
                else:
                    try:
                        # Check if it's a window parameter (typically integer between 2 and 252)
                        val = int(child.value)
                        if val < 2 or val > 252:
                            # Hump uses a float between 0 and 1
                            if node.value != 'hump' and meta.const_args > 0:
                                errors.append(f"Operator '{node.value}' window param {val} is out of bounds [2, 252]")
                    except ValueError:
                        if node.value != 'hump':
                            errors.append(f"Operator '{node.value}' const param '{child.value}' must be an integer")
        
        # Recursively validate children
        if node.children:
            for child in node.children:
                _validate_node(child, registry, valid_fields, errors)
                
    else:
        errors.append(f"Unknown node type: '{node.type}'")
