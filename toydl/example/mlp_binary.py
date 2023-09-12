from toydl.core.optim import SGD
from toydl.core.scalar import Scalar
from toydl.network.mlp import MLPBinaryClassify, MLPConfig


class MLPBinaryClassifyModel:
    def __init__(self, mlp_config: MLPConfig):
        self.model = MLPBinaryClassify(mlp_config)

    def train(self, data, learning_rate, max_epochs=500):
        optim = SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            for i in range(data.n):
                x_1, x_2 = data.X[i]
                y = data.y[i]
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
                (loss / data.n).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


if __name__ == "__main__":
    from toydl.dataset.simulation import datasets

    PTS = 50
    RATE = 0.05
    data = datasets["Simple"](PTS)

    config = MLPConfig(in_size=2, out_size=1, hidden_layer_size=5, hidden_layer_num=2)
    mlp_model = MLPBinaryClassifyModel(config)
    # print(mlp_model.model.order_layer_names)
    mlp_model.train(data, RATE, max_epochs=500)
