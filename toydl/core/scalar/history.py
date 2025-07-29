from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Optional

if TYPE_CHECKING:
    from toydl.core.scalar.scalar import Scalar, ScalarFunction

from dataclasses import dataclass

from toydl.core.scalar.context import Context


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context.py for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional["ScalarFunction"] = None
    ctx: Context | None = None
    inputs: Sequence["Scalar"] = ()
