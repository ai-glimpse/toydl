from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from toydl.core.scalar.scalar import Scalar


def topological_sort(variable: "Scalar") -> Iterable["Scalar"]:
    """
    Computes the topological order of the computation graph.

    :param variable: The right-most variable
    :return: Non-constant Variables in topological order starting from the right.
    """
    variables: list["Scalar"] = []
    visited: set[int] = set()

    def visit(var: "Scalar"):
        if var.is_constant():
            return
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        if var.parents:
            for p in var.parents:
                visit(p)
        variables.append(var)

    visit(variable)
    return variables[::-1]


def backpropagate(variable: "Scalar", deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    :param variable: The right-most variable
    :param deriv: Its derivative that we want to propagate backward to the leaves.

    :return: No return.
    Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
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
