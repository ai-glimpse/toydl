from dataclasses import dataclass


@dataclass
class DatasetValidationError(Exception):
    message: str = "Dataset Validation Error"
