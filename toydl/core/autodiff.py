from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    variables = []
    visited = set()

    def visit(var: Variable):
        if var.is_constant():
            return
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        if var.parents:
            for child_var in var.parents:
                visit(child_var)
        variables.append(var)

    visit(variable)
    return variables[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_values = topological_sort(variable)
    var_derivative_map = {variable.unique_id: deriv}
    for var in sorted_values:
        if var.is_leaf():
            var.accumulate_derivative(var_derivative_map[var.unique_id])
        else:
            var_derivatives = var.chain_rule(var_derivative_map[var.unique_id])
            for _var, _derivative in var_derivatives:
                var_derivative_map[_var.unique_id] = (
                    var_derivative_map.get(_var.unique_id, 0) + _derivative
                )


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
