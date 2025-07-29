from enum import StrEnum
from dataclasses import dataclass

from toydl.core.module import Module
from toydl.core.scalar import Scalar
from toydl.layer.linear import Linear


class ActivationType(StrEnum):
    RELU = "relu"
    SIGMOID = "sigmoid"


@dataclass
class MLPConfig:
    in_size: int
    out_size: int
    hidden_layer_num: int
    hidden_layer_size: int
    hidden_activation: ActivationType = ActivationType.RELU


class MLPBinaryClassifyNetFactory(Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.order_layer_names: list[str] = []
        self.config = self.__check_config(config)
        self.__construct_layers()

    @staticmethod
    def __check_config(config: MLPConfig):
        assert config.out_size == 1, "The out_size in MLPBinaryClassify should be 1"
        return config

    def __construct_layers(self):
        if self.config.hidden_layer_num == 0:
            setattr(
                self,
                "layer_input_output",
                Linear(self.config.in_size, self.config.out_size),
            )
            self.order_layer_names = ["layer_input_output"]
            return

        order_layer_names: list[str] = []
        # input layer
        setattr(
            self,
            "layer_input_hidden_0",
            Linear(self.config.in_size, self.config.hidden_layer_size),
        )
        order_layer_names.append("layer_input_hidden_0")
        # hidden layers: There is already one hidden layer between layer_input and layer_output, so we construct an additional hidden_layer_num - 1 hidden layers here.
        for layer_index in range(self.config.hidden_layer_num - 1):
            middle_layer_name = f"layer_hidden_{layer_index}_{layer_index + 1}"
            setattr(
                self,
                middle_layer_name,
                Linear(self.config.hidden_layer_size, self.config.hidden_layer_size),
            )
            order_layer_names.append(middle_layer_name)

        # output layers
        setattr(
            self,
            "layer_hidden_output",
            Linear(self.config.hidden_layer_size, self.config.out_size),
        )
        order_layer_names.append("layer_hidden_output")

        self.order_layer_names = order_layer_names

    def forward(self, xs: list[Scalar]) -> Scalar:
        for layer_name in self.order_layer_names:
            xs = getattr(self, layer_name).forward(xs)
            if "output" not in layer_name:
                if self.config.hidden_activation == ActivationType.RELU:
                    xs = [h.relu() for h in xs]
                elif self.config.hidden_activation == ActivationType.SIGMOID:
                    xs = [h.sigmoid() for h in xs]
        # note that our output size is 1
        output = xs[0].sigmoid()
        return output


if __name__ == "__main__":
    mlp_config = MLPConfig(
        in_size=2, out_size=1, hidden_layer_size=3, hidden_layer_num=1
    )
    mlp = MLPBinaryClassifyNetFactory(mlp_config)
