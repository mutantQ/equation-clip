"""
Equation Tree Parser for Equation-CLIP Project

Converts LaTeX equations into Operator Tree (OPT) representations
for use with Graph Neural Networks.

Uses SymPy for parsing and canonicalization.
"""

import sympy
from sympy.parsing.latex import parse_latex
from sympy import srepr, latex
import networkx as nx
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Node in the operator tree."""
    node_id: int
    node_type: str  # 'operator', 'symbol', 'number', 'function'
    value: str
    children: List[int]
    parent: Optional[int] = None


@dataclass
class OperatorTree:
    """Operator tree representation of an equation."""
    equation_id: str
    equation_latex: str
    canonical_latex: str
    nodes: List[TreeNode]
    root_id: int
    num_nodes: int
    depth: int


class EquationTreeParser:
    """Parse LaTeX equations into operator trees."""

    # Node type classification
    OPERATORS = {'+', '-', '*', '/', '**', 'Pow', 'Add', 'Mul', 'Div'}
    FUNCTIONS = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Derivative', 'Integral'}

    def __init__(self):
        """Initialize the tree parser."""
        self.node_counter = 0

    def parse_latex_to_sympy(self, latex_str: str) -> Optional[sympy.Expr]:
        """
        Parse LaTeX string to SymPy expression.

        Args:
            latex_str: LaTeX equation string

        Returns:
            SymPy expression or None if parsing fails
        """
        try:
            # Try standard LaTeX parsing
            expr = parse_latex(latex_str)
            return expr
        except Exception as e:
            logger.warning(f"Failed to parse LaTeX: {latex_str[:100]}... Error: {str(e)}")

            # Try manual cleanup and retry
            try:
                # Remove common LaTeX formatting that causes issues
                cleaned = latex_str.replace('\\left', '').replace('\\right', '')
                cleaned = cleaned.replace('\\,', '').replace('\\:', '')
                expr = parse_latex(cleaned)
                return expr
            except:
                return None

    def sympy_to_tree(self, expr: sympy.Expr) -> Tuple[List[TreeNode], int]:
        """
        Convert SymPy expression to operator tree.

        Args:
            expr: SymPy expression

        Returns:
            Tuple of (nodes list, root_id)
        """
        self.node_counter = 0
        nodes = []

        def traverse(expr, parent_id=None):
            """Recursively traverse SymPy expression tree."""
            current_id = self.node_counter
            self.node_counter += 1

            # Determine node type and value
            if isinstance(expr, sympy.Number):
                node_type = 'number'
                value = str(expr)
                children_ids = []
            elif isinstance(expr, sympy.Symbol):
                node_type = 'symbol'
                value = str(expr)
                children_ids = []
            elif isinstance(expr, sympy.Function):
                node_type = 'function'
                value = type(expr).__name__
                children_ids = [traverse(arg, current_id) for arg in expr.args]
            else:
                # Operator node
                node_type = 'operator'
                value = type(expr).__name__
                children_ids = [traverse(arg, current_id) for arg in expr.args]

            # Create node
            node = TreeNode(
                node_id=current_id,
                node_type=node_type,
                value=value,
                children=children_ids,
                parent=parent_id
            )
            nodes.append(node)

            return current_id

        root_id = traverse(expr)
        return nodes, root_id

    def compute_tree_depth(self, nodes: List[TreeNode], root_id: int) -> int:
        """Compute depth of the tree."""
        def get_depth(node_id):
            node = nodes[node_id]
            if not node.children:
                return 1
            return 1 + max(get_depth(child_id) for child_id in node.children)

        return get_depth(root_id)

    def parse_equation(self, equation_latex: str, equation_id: str) -> Optional[OperatorTree]:
        """
        Parse equation to operator tree.

        Args:
            equation_latex: LaTeX equation string
            equation_id: Unique identifier for equation

        Returns:
            OperatorTree object or None if parsing fails
        """
        try:
            # Parse LaTeX to SymPy
            expr = self.parse_latex_to_sympy(equation_latex)
            if expr is None:
                return None

            # Canonicalize expression
            try:
                canonical_expr = sympy.simplify(expr)
                canonical_latex = latex(canonical_expr)
            except:
                canonical_expr = expr
                canonical_latex = equation_latex

            # Convert to tree
            nodes, root_id = self.sympy_to_tree(canonical_expr)

            # Compute tree depth
            depth = self.compute_tree_depth(nodes, root_id)

            # Create operator tree
            op_tree = OperatorTree(
                equation_id=equation_id,
                equation_latex=equation_latex,
                canonical_latex=canonical_latex,
                nodes=nodes,
                root_id=root_id,
                num_nodes=len(nodes),
                depth=depth
            )

            return op_tree

        except Exception as e:
            logger.error(f"Error parsing equation {equation_id}: {str(e)}")
            return None

    def tree_to_networkx(self, op_tree: OperatorTree) -> nx.DiGraph:
        """
        Convert operator tree to NetworkX graph for visualization/GNN.

        Args:
            op_tree: OperatorTree object

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes
        for node in op_tree.nodes:
            G.add_node(
                node.node_id,
                node_type=node.node_type,
                value=node.value
            )

        # Add edges (parent -> child)
        for node in op_tree.nodes:
            for child_id in node.children:
                G.add_edge(node.node_id, child_id)

        return G

    def parse_batch(self, equations: List[Dict], output_file: Path) -> List[OperatorTree]:
        """
        Parse a batch of equations to operator trees.

        Args:
            equations: List of equation dictionaries
            output_file: Output file to save parsed trees

        Returns:
            List of successfully parsed OperatorTree objects
        """
        parsed_trees = []
        failed_count = 0

        for eq_dict in equations:
            equation_id = eq_dict['equation_id']
            equation_latex = eq_dict['equation_latex']

            op_tree = self.parse_equation(equation_latex, equation_id)

            if op_tree is not None:
                parsed_trees.append(op_tree)
            else:
                failed_count += 1

        # Save parsed trees
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            tree_dicts = []
            for tree in parsed_trees:
                tree_dict = asdict(tree)
                # Convert TreeNode objects to dicts
                tree_dict['nodes'] = [asdict(node) for node in tree.nodes]
                tree_dicts.append(tree_dict)
            json.dump(tree_dicts, f, indent=2)

        logger.info(f"Parsed {len(parsed_trees)} equations successfully")
        logger.info(f"Failed to parse {failed_count} equations")

        # Statistics
        if parsed_trees:
            avg_nodes = sum(tree.num_nodes for tree in parsed_trees) / len(parsed_trees)
            avg_depth = sum(tree.depth for tree in parsed_trees) / len(parsed_trees)
            logger.info(f"Average nodes per tree: {avg_nodes:.1f}")
            logger.info(f"Average tree depth: {avg_depth:.1f}")

        return parsed_trees


def main():
    """Main execution function."""
    # Configuration
    INPUT_FILE = Path('./data/extracted_equations.json')
    OUTPUT_FILE = Path('./data/equation_trees.json')

    # Load extracted equations
    logger.info(f"Loading equations from {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        equations = json.load(f)

    logger.info(f"Loaded {len(equations)} equations")

    # Parse to trees
    parser = EquationTreeParser()
    logger.info("Parsing equations to operator trees...")
    parsed_trees = parser.parse_batch(equations, OUTPUT_FILE)

    logger.info(f"Parsing complete. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
