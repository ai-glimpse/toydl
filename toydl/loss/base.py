from dataclasses import dataclass


@dataclass(frozen=True, repr=True)
class LossBase:
    name: str = "Base Loss"
