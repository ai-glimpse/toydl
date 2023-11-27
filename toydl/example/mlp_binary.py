import matplotlib.pyplot as plt

from toydl.core.optim import SGD, Momentum, Optimizer
from toydl.core.scalar import Scalar
from toydl.network.mlp import MLPBinaryClassify, MLPConfig


class MLPBinaryClassifyModel:
    def __init__(self, mlp_config: MLPConfig):
        self.model = MLPBinaryClassify(mlp_config)

    def evaluate(self, dateset):
        n = len(dateset)
        total_loss = 0.0
        correct = 0
        for i in range(n):
            x_1, x_2 = dateset.X[i]
            y = dateset.y[i]
            x_1 = Scalar(x_1)
            x_2 = Scalar(x_2)
            out = self.model.forward((x_1, x_2))
            if y == 1:
                prob = out
                correct += 1 if out.data > 0.5 else 0
            else:
                prob = -out + 1.0
                correct += 1 if out.data < 0.5 else 0
            loss = -prob.log()
            total_loss += loss.data

        return total_loss, correct

    def train(self, training_set, test_set, optimizer: Optimizer, max_epochs=500):
        training_loss, testing_loss = [], []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optimizer.zero_grad()

            # Forward
            for i in range(training_set.n):
                x_1, x_2 = training_set.X[i]
                y = training_set.y[i]
                x_1 = Scalar(x_1)
                x_2 = Scalar(x_2)
                out = self.model.forward((x_1, x_2))
                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / training_set.n).backward()
                total_loss += loss.data
            training_loss.append(total_loss)

            # Update
            optimizer.step()

            test_total_loss, test_correct = self.evaluate(test_set)
            testing_loss.append(test_total_loss)
            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                print("Epoch ", epoch, "train loss ", total_loss, "correct", correct)
                print("Test loss ", test_total_loss, "correct", test_correct)
        return training_loss, testing_loss

    def plot_loss(self, training_loss, testing_loss):
        plt.plot(training_loss, "ko-")
        plt.plot(testing_loss, "g*-")
        plt.show()


if __name__ == "__main__":
    from toydl.dataset.simulation import datasets

    PTS = 300
    learning_rate = 0.1
    momentum = 0.9
    max_epochs = 100

    data = datasets["Xor"](PTS)
    training_set, test_set = data.train_test_split(train_proportion=0.7)

    config = MLPConfig(in_size=2, out_size=1, hidden_layer_size=10, hidden_layer_num=2)
    mlp_model = MLPBinaryClassifyModel(config)
    sgd_optimizer = SGD(mlp_model.model.parameters(), learning_rate)
    momentum_optimizer = Momentum(mlp_model.model.parameters(), learning_rate, momentum)
    # print(mlp_model.model.order_layer_names)
    training_loss, testing_loss = mlp_model.train(
        training_set, test_set, momentum_optimizer, max_epochs=max_epochs
    )
    mlp_model.plot_loss(training_loss, testing_loss)
