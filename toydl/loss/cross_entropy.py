from dataclasses import dataclass, field

from toydl.core.scalar import Scalar
from toydl.loss.base import LossBase


@dataclass(frozen=True, repr=True)
class CrossEntropyLoss(LossBase):
    name: str = field(default="Cross Entropy Loss")

    @staticmethod
    def forward(y_true: int, y_pred: Scalar) -> Scalar:
        return -1 * ((y_true * y_pred.log()) + (1 - y_true) * ((1 - y_pred).log()))
