import random
import gc
import math
import time


class Tensor:
    def __init__(self, value, parents=set(), grad=None):
        self.value = value
        self.parents = parents
        self._backward = lambda: None
        self.grad = 0

    def __repr__(self):
        return f"Tensor <{self.value}>"

    def __add__(self, x2):
        if isinstance(x2, Tensor):
            y = Tensor(self.value + x2.value, (self, x2))
        else:
            x2 = Tensor(x2)
            y = Tensor(self.value + x2.value, (self, x2))

        def backward():
            op_grad = 1.0
            chain_rule = y.grad
            self.grad += op_grad * (chain_rule)
            x2.grad += op_grad * (chain_rule)

        y._backward = backward
        return y

    def __sub__(self, x2):
        return self + (-x2)

    def __neg__(self):  # -self
        return self * -1

    def __rsub__(self, x2):
        return x2 + (-self)

    def __mul__(self, x2):
        if isinstance(x2, Tensor):
            y = Tensor(self.value * x2.value, (self, x2))
        else:
            x2 = Tensor(x2)
            y = Tensor(self.value * x2.value, (self, x2))

        def backward():
            self.grad += x2.value * y.grad
            x2.grad += self.value * y.grad

        y._backward = backward
        return y

    def __pow__(self, power):
        y = Tensor(self.value**power, (self,))

        def backward():
            self.grad += (power * self.value ** (power - 1)) * y.grad

        y._backward = backward
        return y

    def relu(self):
        y = Tensor(0 if self.value < 0 else self.value, (self,))

        def _backward():
            self.grad += (y.value > 0) * y.grad

        y._backward = _backward

        return y

    def tanh(self):
        x = self.value
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        y = Tensor(t, (self,))

        def _backward():
            self.grad += (1 - t**2) * y.grad

        y._backward = _backward

        return y

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [
            [Tensor(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)
        ]
        self.bias = [Tensor(random.uniform(-1, 1)) for _ in range(nout)]

    def __call__(self, inputs):
        out = []
        for neuron, b in zip(self.neurons, self.bias):
            y = sum(((wi * xi).tanh() for wi, xi in zip(neuron, inputs)), b)
            out.append(y)
        return out


class FFN:
    def __init__(self):
        self.layers = [
            Layer(3, 30),
            Layer(30, 1),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]

    def zero_grad(self):
        for layer in self.layers:
            for i in range(len(layer.neurons)):
                for j in range(len(layer.neurons[i])):
                    layer.neurons[i][j].grad = 0
            for i in range(len(layer.bias)):
                layer.bias[i].grad = 0

    def sgd(self, step):
        for layer in self.layers:
            for i in range(len(layer.neurons)):
                for j in range(len(layer.neurons[i])):
                    layer.neurons[i][j].value += -layer.neurons[i][j].grad * step
            for i in range(len(layer.bias)):
                layer.bias[i].value += -layer.bias[i].grad * step


import gc
import sys
import time
import io


def capture_gc_time():
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    gc.collect()
    debug_output = sys.stderr.getvalue()
    sys.stderr = old_stderr

    total_gc_time = 0.0
    for line in debug_output.splitlines():
        if "elapsed" in line:
            parts = line.split(",")
            for part in parts:
                if "elapsed" in part:
                    time_str = part.split()[0]
                    total_gc_time += float(time_str[:-1])
    return total_gc_time


if __name__ == "__main__":
    initial_gc_time = capture_gc_time()

    start_time = time.time()
    inputs = [
        [2, 3, -1],
        [-1, 0, -2],
        [3, 2, 3],
    ]

    ys = [
        0,
        -1,
        1,
    ]
    n = FFN()

    epochs = 100
    for epoch in range(epochs):
        n.zero_grad()
        preds = [n(x) for x in inputs]
        loss = sum(((ypred - yout) ** 2 for ypred, yout in zip(preds, ys)), Tensor(0))
        loss.backward()
        n.sgd(0.01)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.value}")

    print(f"Epoch: {epoch}, Loss: {loss.value}")
    end_time = time.time()
    print(preds)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    final_gc_time = capture_gc_time()

    gc_time_elapsed = final_gc_time - initial_gc_time

    print(f"Total time elapsed by garbage collector: {gc_time_elapsed} seconds")
