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


class MLPBinaryClassifyNetFactory(Module):
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
        if self.config.hidden_layer_num == 0:
            setattr(
                self,
                "layer_input_output",
                Linear(self.config.in_size, self.config.out_size),
            )
            self.order_layer_names = ["layer_input_output"]
            return

        order_layer_names = []
        # input layer
        setattr(
            self,
            "layer_input_hidden_0",
            Linear(self.config.in_size, self.config.hidden_layer_size),
        )
        order_layer_names.append("layer_input_hidden_0")
        # hidden layers: 目前 layer_input 和 layer_output 之间已经有了一个隐层，所以这里构建额外 hidden_layer_num - 1 个隐层
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

    def forward(self, x) -> Scalar:
        for layer_name in self.order_layer_names:
            x = getattr(self, layer_name).forward(x)
            if "output" not in layer_name:
                x = [h.relu() for h in x]
        x = x[0].sigmoid()
        return x


if __name__ == "__main__":
    from toydl.core.scalar import Scalar
    from toydl.core.scalar.bp import topological_sort, backpropagate  # noqa: F401

    mlp_config = MLPConfig(
        in_size=2, out_size=1, hidden_layer_size=3, hidden_layer_num=1
    )
    mlp = MLPBinaryClassifyNetFactory(mlp_config)
    for p in mlp.named_parameters():
        print(p)

    x1 = Scalar(1.0, name="x1")
    x2 = Scalar(2.0, name="x2")
    xs = (x1, x2)
    y = mlp.forward(xs)
    y.name = "y"

    y.backward()

    print("After backward:")
    for p in mlp.named_parameters():
        print(p)
