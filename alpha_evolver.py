"""
Alpha Evolver: Genetic algorithm performing crossover and mutation directly on AST trees.
Maintains a population of high-performing alphas, evolving them across generations.
"""
import os
import random
import polars as pl
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from ast_node import ASTNode, parse_formula
from operator_registry import OperatorRegistry, SUPPORTED_OPS
from alpha_validator import validate_ast
from alpha_templates import AlphaTemplateGenerator

@dataclass
class Individual:
    ast: ASTNode
    formula: str
    fitness: float
    metrics: Dict[str, Any]


class AlphaEvolver:
    """
    AST-based Genetic Programming optimizer for WorldQuant Brain alphas.
    """

    def __init__(
        self,
        registry: OperatorRegistry,
        valid_fields: Optional[List[str]] = None,
        population_size: int = 40,
        max_depth: int = 6,
        crossover_rate: float = 0.70,
        mutation_rate: float = 0.35,
        elite_pct: float = 0.15
    ):
        self.registry = registry
        self.valid_fields = valid_fields or ["open", "high", "low", "close", "volume"]
        self.population_size = population_size
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_pct = elite_pct

        self.population: List[Individual] = []
        self.template_generator = AlphaTemplateGenerator(self.valid_fields)

        # Pre-group operators by arity for safe operator mutations
        self._op_by_arity: Dict[Tuple[int, int], List[str]] = {}
        for op_name, spec in SUPPORTED_OPS.items():
            meta = self.registry.get_operator(op_name)
            if meta and meta.has_polars_impl and meta.level == "ALL" and "REGULAR" in meta.scope:
                key = (spec['exprs'], spec['consts'])
                self._op_by_arity.setdefault(key, []).append(op_name)

    def calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """
        Computes composite multi-objective fitness:
        - Rewards median Sharpe and consistency
        - Rewards worst-window drawdown resilience
        - Penalizes high turnover
        """
        sharpe = float(metrics.get('median_sharpe') or 0.0)
        worst_sharpe = float(metrics.get('worst_sharpe') or -99.0)
        consistency = float(metrics.get('consistency') or 0.0)
        turnover = float(metrics.get('turnover') or 0.0)

        # Base fitness
        fit = (sharpe * 2.0) + ((consistency / 100.0) * 1.5) + (max(-2.0, worst_sharpe) * 0.5)

        # Penalize turnover > 60%
        if turnover > 60.0:
            fit -= ((turnover - 60.0) / 10.0) * 0.2

        # Massive penalty for degenerate or zero values
        if sharpe == 0.0 and consistency == 0.0:
            fit = -10.0

        return fit

    def seed_from_logs(self, logs_csv_path: str = "logs_processed.csv", top_n: int = 20) -> int:
        """
        Seeds population with the highest-performing alphas from processed logs.
        """
        if not os.path.exists(logs_csv_path):
            return 0

        try:
            df = pl.read_csv(logs_csv_path)
            if len(df) == 0 or 'formula' not in df.columns:
                return 0

            # Filter valid, positive Sharpe rows
            filtered = df.filter(pl.col('median_sharpe') > 0.3)
            if len(filtered) == 0:
                filtered = df

            # Sort by median_sharpe descending
            sorted_df = filtered.sort('median_sharpe', descending=True).head(top_n * 2)

            loaded = 0
            for row in sorted_df.iter_rows(named=True):
                formula = row.get('formula')
                if not formula or not isinstance(formula, str):
                    continue

                try:
                    ast = parse_formula(formula)
                    is_valid, _ = validate_ast(ast, self.registry, self.valid_fields)
                    if not is_valid:
                        continue

                    metrics = {
                        'median_sharpe': float(row.get('median_sharpe') or 0.0),
                        'worst_sharpe': float(row.get('worst_sharpe') or 0.0),
                        'consistency': float(row.get('consistency') or 0.0),
                        'turnover': float(row.get('turnover') or 0.0)
                    }
                    fit = self.calculate_fitness(metrics)
                    self.add_individual(ast, metrics, fit)
                    loaded += 1
                    if loaded >= top_n:
                        break
                except Exception:
                    continue

            return loaded
        except Exception:
            return 0

    def seed_from_templates(self, count: int = 20):
        """Seeds population with diverse template alphas."""
        for _ in range(count):
            ast = self.template_generator.generate(apply_wrappers=True)
            is_valid, _ = validate_ast(ast, self.registry, self.valid_fields)
            if is_valid:
                # Initial default metrics for un-evaluated templates
                metrics = {'median_sharpe': 0.5, 'worst_sharpe': 0.0, 'consistency': 50.0, 'turnover': 20.0}
                fit = self.calculate_fitness(metrics)
                self.add_individual(ast, metrics, fit)

    def add_individual(self, ast: ASTNode, metrics: Dict[str, Any], fitness: Optional[float] = None):
        """Adds an individual to the population and keeps population sorted by fitness."""
        formula = ast.to_string()
        # Avoid duplicate formulas
        for ind in self.population:
            if ind.formula == formula:
                # Update metrics if better
                if fitness is not None and fitness > ind.fitness:
                    ind.fitness = fitness
                    ind.metrics = metrics
                return

        fit = fitness if fitness is not None else self.calculate_fitness(metrics)
        self.population.append(Individual(ast=ast, formula=formula, fitness=fit, metrics=metrics))
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Trim excess individuals beyond population_size
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]

    def select_parent(self, tournament_k: int = 3) -> ASTNode:
        """Tournament selection: picks k candidates at random, returns the fittest."""
        if not self.population:
            return self.template_generator.generate(apply_wrappers=True)

        k = min(tournament_k, len(self.population))
        contenders = random.sample(self.population, k)
        winner = max(contenders, key=lambda x: x.fitness)
        return winner.ast.clone()

    def crossover(self, parent1: ASTNode, parent2: ASTNode) -> Tuple[ASTNode, ASTNode]:
        """
        Subtree Crossover:
        Selects a random non-root subtree in parent1 and swaps it with a compatible subtree in parent2.
        """
        c1 = parent1.clone()
        c2 = parent2.clone()

        subtrees1 = c1.collect_subtrees_with_context()
        subtrees2 = c2.collect_subtrees_with_context()

        # Filter to subtrees that have parents (non-root) to swap
        swappable1 = [(p, idx, node) for p, idx, node in subtrees1 if p is not None and node.type in ('operator', 'field')]
        swappable2 = [(p, idx, node) for p, idx, node in subtrees2 if p is not None and node.type in ('operator', 'field')]

        if not swappable1 or not swappable2:
            return c1, c2

        p1, idx1, node1 = random.choice(swappable1)
        p2, idx2, node2 = random.choice(swappable2)

        # Perform subtree swap
        p1.children[idx1] = node2.clone()
        p2.children[idx2] = node1.clone()

        # If child exceeds max_depth, revert to original parent
        if c1.depth() > self.max_depth:
            c1 = parent1.clone()
        if c2.depth() > self.max_depth:
            c2 = parent2.clone()

        return c1, c2

    def mutate(self, ast: ASTNode) -> ASTNode:
        """
        Applies a random mutation to the AST tree:
        1. Operator swap with same arity (e.g. ts_mean <-> ts_decay_linear)
        2. Window constant perturbation (e.g. 10 -> 15)
        3. Field swap (e.g. close <-> open)
        4. Subtree replacement with small template subtree
        """
        cloned = ast.clone()
        mutation_type = random.choice(['op_swap', 'window_tweak', 'field_swap', 'subtree_replace'])

        if mutation_type == 'op_swap':
            # Swap an operator with another of same arity
            op_nodes = [n for n in cloned.collect_nodes() if n.type == 'operator']
            if op_nodes:
                target = random.choice(op_nodes)
                spec = SUPPORTED_OPS.get(target.value)
                if spec:
                    key = (spec['exprs'], spec['consts'])
                    compatible = [op for op in self._op_by_arity.get(key, []) if op != target.value]
                    if compatible:
                        target.value = random.choice(compatible)
                        return cloned

        elif mutation_type == 'window_tweak':
            # Tweak integer constant parameters
            const_nodes = [n for n in cloned.collect_nodes() if n.type == 'constant']
            if const_nodes:
                target = random.choice(const_nodes)
                try:
                    val = int(target.value)
                    delta = random.choice([-5, -2, 2, 5, 10])
                    new_val = max(2, min(252, val + delta))
                    target.value = str(new_val)
                    return cloned
                except ValueError:
                    pass

        elif mutation_type == 'field_swap':
            # Swap a field with another valid field
            field_nodes = [n for n in cloned.collect_nodes() if n.type == 'field']
            if field_nodes and len(self.valid_fields) > 1:
                target = random.choice(field_nodes)
                alt_fields = [f for f in self.valid_fields if f != target.value]
                if alt_fields:
                    target.value = random.choice(alt_fields)
                    return cloned

        # Fallback or subtree_replace: replace a random child with a fresh small template subtree
        subtrees = cloned.collect_subtrees_with_context()
        swappable = [(p, idx, node) for p, idx, node in subtrees if p is not None and node.type == 'operator']
        if swappable:
            p, idx, _ = random.choice(swappable)
            fresh_subtree = self.template_generator.generate(apply_wrappers=False)
            p.children[idx] = fresh_subtree

        if cloned.depth() > self.max_depth:
            return ast.clone()

        return cloned

    def generate_batch(self, count: int = 5) -> List[ASTNode]:
        """
        Generates a batch of candidate ASTNodes using genetic evolution:
        - Elitism
        - Subtree Crossover
        - Mutation
        """
        if len(self.population) < 2:
            # Not enough individuals, seed more from templates
            self.seed_from_templates(count=20)

        candidates: List[ASTNode] = []
        attempts = 0
        max_attempts = count * 6

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            roll = random.random()

            if roll < self.crossover_rate:
                # Crossover
                p1 = self.select_parent()
                p2 = self.select_parent()
                offspring1, offspring2 = self.crossover(p1, p2)
                chosen = random.choice([offspring1, offspring2])
            else:
                # Mutation
                p = self.select_parent()
                chosen = self.mutate(p)

            # Optional additional mutation
            if random.random() < self.mutation_rate:
                chosen = self.mutate(chosen)

            # Validate before accepting
            is_valid, _ = validate_ast(chosen, self.registry, self.valid_fields)
            if is_valid and chosen.depth() <= self.max_depth:
                candidates.append(chosen)

        # If not enough, fill with templates
        while len(candidates) < count:
            tmpl = self.template_generator.generate(apply_wrappers=True)
            is_valid, _ = validate_ast(tmpl, self.registry, self.valid_fields)
            if is_valid:
                candidates.append(tmpl)

        return candidates
