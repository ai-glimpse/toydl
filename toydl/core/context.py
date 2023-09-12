from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class Context:
    """
    Context class is used by `ScalarFunction` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        :param values: the values that should be saved for backward
        """
        if self.no_grad:
            return
        self.saved_values = values
