import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


if __name__ == "__main__":
    start_time = time.time()  # Start timing
    inputs = np.array(
        [
            [-2, 1, 0, -1, 2, 1, -1, 2, -2, 1],
            [1, -1, 2, -2, 1, 0, 2, -1, 0, -3],
            [0, -2, 1, -1, 2, -1, 0, -1, 1, -3],
            [-1, 2, -1, 1, -2, 0, -1, 2, 0, 1],
            [2, -1, 1, -2, 1, 2, -1, 0, 1, -1],
            [0, 1, -2, 2, -1, 1, -1, 0, 2, -1],
        ]
    )

    ys = np.array([1, -2, 2, -1, 1, -2]).reshape(
        -1, 1
    )  # Reshape ys to be a column vector

    # Convert inputs and ys to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)

    model = FFN()
    loss_fn = F.mse_loss
    opt = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        opt.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, ys)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    end_time = time.time()  # End timing
    print(f"Total training time: {end_time - start_time:.2f} seconds")
