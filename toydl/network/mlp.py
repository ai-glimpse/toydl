from dataclasses import dataclass
from typing import List

from toydl.core.module import Module
from toydl.core.scalar import Scalar
from toydl.layer.linear import Linear


@dataclass
class MLPConfig:
    in_size: int
    out_size: int
    hidden_layer_num: int
    hidden_layer_size: int


class MLPBinaryClassify(Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.order_layer_names: List[str] = []
        self.config = self.__check_config(config)
        self.__construct_layers()

    @staticmethod
    def __check_config(config: MLPConfig):
        assert config.out_size == 1, "The out_size in MLPBinaryClassify should be 1"
        return config

    def __construct_layers(self):
        order_layer_names = []
        # input layer
        setattr(
            self,
            "layer_input",
            Linear(self.config.in_size, self.config.hidden_layer_size),
        )
        order_layer_names.append("layer_input")
        # hidden layers
        for layer_index in range(self.config.hidden_layer_num):
            middle_layer_name = f"layer_hidden_{layer_index + 1}"
            setattr(
                self,
                middle_layer_name,
                Linear(self.config.hidden_layer_size, self.config.hidden_layer_size),
            )
            order_layer_names.append(middle_layer_name)

        # output layers
        setattr(
            self,
            "layer_output",
            Linear(self.config.hidden_layer_size, self.config.out_size),
        )
        order_layer_names.append("layer_output")

        self.order_layer_names = order_layer_names

    def forward(self, x) -> Scalar:
        for layer_name in self.order_layer_names:
            x = getattr(self, layer_name).forward(x)
            if layer_name != "layer_output":
                x = [h.relu() for h in x]
        x = x[0].sigmoid()
        return x


if __name__ == "__main__":
    mlp_config = MLPConfig(
        in_size=2, out_size=1, hidden_layer_size=10, hidden_layer_num=5
    )
    mlp = MLPBinaryClassify(mlp_config)
