import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity


def total_flops(input_size, layers):
    def flops(n, m, p):
        # np(2n-1)
        # nXm
        # mXp
        matmul = (m * p) * ((2 * n) - 1)
        bias = m * p
        return matmul  # + bias

    n = len(input_size[0])
    total_flops = 0

    for layer in layers:
        m, p = layer
        total_flops += flops(n, m, p)
        # TODO Do I need to update anything right here?

    return total_flops


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(3, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
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
            [2, 3, -1],
            [-1, 0, -2],
            [3, 2, 3],
        ]
    )

    ys = np.array([0, -1, 1]).reshape(-1, 1)  # Reshape ys to be a column vector

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
    print(out)

    model_layers = [
        [3, 125],
        [125, 3500],
        [3500, 3500],
        [3500, 3500],
        [3500, 125],
        [125, 1],
    ]

    # Input size (list of lists)
    input_size = [[2, 3, -1], [-1, 0, -2], [3, 2, 3]]

    # Calculate total FLOPS
    total_flops = total_flops(input_size, model_layers)
    print(f"Total FLOPS: {total_flops}")

    end_time = time.time()  # End timing
    print(f"Execution time: {end_time - start_time:.2f} seconds")
