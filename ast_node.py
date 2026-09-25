from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ASTNode:
    type: str  # 'field', 'constant', 'operator'
    value: str 
    children: Optional[List['ASTNode']] = None

    def to_string(self) -> str:
        if self.type in ('field', 'constant'):
            return str(self.value)
        elif self.type == 'operator':
            args = ", ".join(child.to_string() for child in (self.children or []))
            return f"{self.value}({args})"
        return ""

    def clone(self) -> 'ASTNode':
        """Deep copy of the ASTNode and all descendant nodes."""
        cloned_children = [c.clone() for c in self.children] if self.children is not None else None
        return ASTNode(type=self.type, value=self.value, children=cloned_children)

    def depth(self) -> int:
        """Returns maximum depth of the tree rooted at this node (leaf depth = 1)."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """Returns total count of nodes in the tree."""
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)

    def collect_nodes(self) -> List['ASTNode']:
        """Returns a flat list of all nodes in this subtree (pre-order)."""
        nodes = [self]
        if self.children:
            for c in self.children:
                nodes.extend(c.collect_nodes())
        return nodes

    def collect_subtrees_with_context(self) -> List[Tuple[Optional['ASTNode'], int, 'ASTNode']]:
        """
        Returns list of (parent, child_index, node) tuples for all subtrees.
        Root has parent=None, child_index=-1.
        Essential for subtree replacement during crossover and mutation.
        """
        result = [(None, -1, self)]
        def _traverse(parent: 'ASTNode'):
            if parent.children:
                for idx, child in enumerate(parent.children):
                    result.append((parent, idx, child))
                    _traverse(child)
        _traverse(self)
        return result


def parse_formula(s: str) -> ASTNode:
    """
    Parses a formula string (e.g. 'ts_decay_linear(rank(close), 10)') back into an ASTNode tree.
    Supports nested operators, fields, integer/float constants, and whitespace.
    """
    s = s.strip()
    if not s:
        raise ValueError("Cannot parse empty formula string")

    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] in '(),':
            tokens.append(s[i])
            i += 1
        elif s[i] == '-' and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
            # Negative number constant
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i].isalnum() or s[i] in '._':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] in '._'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"Unexpected character in formula: '{s[i]}' at position {i}")

    pos = 0
    def _parse_expr() -> ASTNode:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of formula while parsing")
        tok = tokens[pos]
        pos += 1

        # Operator call: token followed by '('
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1  # consume '('
            children = []
            if pos < len(tokens) and tokens[pos] != ')':
                while True:
                    children.append(_parse_expr())
                    if pos < len(tokens) and tokens[pos] == ',':
                        pos += 1  # consume ','
                    else:
                        break
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError(f"Missing closing ')' for operator '{tok}'")
            pos += 1  # consume ')'
            return ASTNode(type='operator', value=tok, children=children)
        else:
            # Leaf node: numeric constant or field identifier
            try:
                float(tok)
                return ASTNode(type='constant', value=tok)
            except ValueError:
                return ASTNode(type='field', value=tok)

    root = _parse_expr()
    if pos < len(tokens):
        raise ValueError(f"Trailing tokens found after parsing formula: {tokens[pos:]}")
    return root

