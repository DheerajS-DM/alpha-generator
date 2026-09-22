from dataclasses import dataclass
from typing import List

@dataclass
class ASTNode:
    type: str  
    value: str 
    children: List['ASTNode'] = None

    def to_string(self) -> str:
        if self.type in ('field', 'constant'):
            return str(self.value)
        elif self.type == 'operator':
            args = ", ".join(child.to_string() for child in (self.children or []))
            return f"{self.value}({args})"
        return ""
