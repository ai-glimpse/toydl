from typing import List, Tuple

import matplotlib.pyplot as plt

import toydl.dataset.simulation as simulation_dataset

from toydl.core.optim import SGD, Momentum, Optimizer
from toydl.core.scalar.scalar import Scalar
from toydl.dataset.simple import SimpleDataset
from toydl.loss.cross_entropy import CrossEntropyLoss
from toydl.network.mlp import MLPBinaryClassifyNetFactory, MLPConfig


# --8<-- [start:model]
class MLPBinaryClassifyModel:
    def __init__(self, mlp_config: MLPConfig):
        self.net = MLPBinaryClassifyNetFactory(mlp_config)

    def forward_once(self, x: list[float], y: int) -> tuple[Scalar, Scalar]:
        y_pred = self.net.forward(list(Scalar(v) for v in x))
        loss = CrossEntropyLoss().forward(y_true=y, y_pred=y_pred)
        return y_pred, loss

    def evaluate(self, dateset: SimpleDataset) -> tuple[float, int]:
        # switch to eval mode
        self.net.eval()
        total_loss = 0.0
        correct = 0
        for x, y in dateset:
            y_pred, loss = self.forward_once(x, y)
            if y == 1:
                correct += 1 if y_pred.data > 0.5 else 0
            else:
                correct += 1 if y_pred.data < 0.5 else 0
            total_loss += loss.data

        # switch back to train mode
        self.net.train()

        return total_loss, correct

    def train(
        self,
        training_set: SimpleDataset,
        test_set: SimpleDataset,
        optimizer: Optimizer,
        max_epochs: int = 500,
    ) -> Tuple[List[float], List[float], str]:
        training_loss, testing_loss = [], []
        for epoch in range(1, max_epochs + 1):
            optimizer.zero_grad()

            # Forward & Backward
            for x, y in training_set:
                _, loss = self.forward_once(x, y)
                (loss / len(training_set)).backward()

            # Update parameters
            optimizer.step()

            # Evaluation
            train_loss, train_correct = self.evaluate(training_set)
            test_loss, test_correct = self.evaluate(test_set)

            training_loss.append(train_loss)
            testing_loss.append(test_loss)
            if epoch % 10 == 0 or epoch == max_epochs:
                print(
                    f"[Epoch {epoch}]Train Loss = {train_loss}, "
                    f"right({train_correct})/total({len(training_set)}) = {train_correct / len(training_set)}\n"
                    f"[Epoch {epoch}]Test  Loss = {test_loss},  "
                    f"right({test_correct})/total({len(test_set)}) = {test_correct / len(test_set)}"
                )
        test_result = f"right/total = {test_correct}/{len(test_set)}"
        return training_loss, testing_loss, test_result

    # --8<-- [end:model]

    @staticmethod
    def plot_loss(
        training_loss: List[float],
        testing_loss: List[float],
        title: str = "loss plot",
        filename: str = "loss.png",
    ):
        plt.clf()
        plt.plot(training_loss, "ro-", label="training loss")
        plt.plot(testing_loss, "g*-", label="test loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()


# --8<-- [start:gen_dateset]
def get_dataset(n: int = 100) -> Tuple[SimpleDataset, SimpleDataset]:
    data = simulation_dataset.simple(n)
    training_set, test_set = data.train_test_split(train_proportion=0.7)

    return training_set, test_set


# --8<-- [end:gen_dateset]


def run_sgd(
    mlp_config: MLPConfig,
    training_set: SimpleDataset,
    test_set: SimpleDataset,
    learning_rate: float,
    max_epochs: int = 500,
):
    mlp_model = MLPBinaryClassifyModel(mlp_config)

    sgd_optimizer = SGD(mlp_model.net.parameters(), learning_rate)

    training_loss, testing_loss, test_result = mlp_model.train(
        training_set, test_set, sgd_optimizer, max_epochs=max_epochs
    )
    mlp_model.plot_loss(
        training_loss, testing_loss, title=f"SGD: {test_result}", filename="sgd.png"
    )


def run_momentum(
    mlp_config: MLPConfig,
    training_set: SimpleDataset,
    test_set: SimpleDataset,
    learning_rate: float,
    max_epochs: int = 500,
):
    momentum = 0.5
    mlp_model = MLPBinaryClassifyModel(mlp_config)

    optimizer = Momentum(mlp_model.net.parameters(), learning_rate, momentum)

    training_loss, testing_loss, test_result = mlp_model.train(
        training_set, test_set, optimizer, max_epochs=max_epochs
    )
    mlp_model.plot_loss(
        training_loss,
        testing_loss,
        title=f"Momentum: {test_result}",
        filename="momentum.png",
    )


def run():
    n = 100
    training_set, test_set = get_dataset(n)
    training_set.plot(filename="training_set.png")
    mlp_config = MLPConfig(
        in_size=2, out_size=1, hidden_layer_size=10, hidden_layer_num=2
    )

    learning_rate = 0.01
    max_epochs = 500

    run_sgd(mlp_config, training_set, test_set, learning_rate, max_epochs)
    run_momentum(mlp_config, training_set, test_set, learning_rate, max_epochs)


if __name__ == "__main__":
    run()
