import random


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
            self.grad += 1.0 * (y.grad)
            x2.grad += 1.0 * (y.grad)

        y._backward = backward
        return y

    def __sub__(self, x2):
        if isinstance(x2, Tensor):
            y = Tensor(self.value - x2.value, (self, x2))
        else:
            x2 = Tensor(x2)
            y = Tensor(self.value - x2.value, (self, x2))

        def backward():
            self.grad += 1.0 * (y.grad)
            x2.grad += 1.0 * (y.grad)

        y._backward = backward
        return y

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

    def leaky_relu(self):
        if self.value > 0:
            y = Tensor(self.value, (self,))
        else:
            y = Tensor(self.value * 0.01, (self,))

        def backward():
            # NOTE: 1 if x>=0 else 0.01
            self.grad += 1 if self.value >= 0 else 0.01 * y.grad

        y._backward = backward

        return y

    def backward(self):
        # NOTE: Sorting topologically is necessary so that we can call the backwards pass in order. If we do it out of order we would be
        # mulyiplying the wrong things for the chain rule
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [
            [Tensor(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)
        ]
        self.bias = [Tensor(0) for _ in range(nout)]

    def __call__(self, inputs):
        out = []
        for neuron, b in zip(self.neurons, self.bias):
            y = sum(((wi * xi).leaky_relu() for wi, xi in zip(neuron, inputs)), b)
            out.append(y)
        return out


class FFN:
    def __init__(self):
        self.layers = [Layer(3, 16), Layer(16, 8), Layer(8, 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]


if __name__ == "__main__":
    inputs = [[0.5, 1, 0.2], [0.2, 0.3, 0.4], [1, 0.8, 0.9]]
    ys = [1.0, 1.0, -1.0]
    n = FFN()
    pred = [n(x) for x in inputs]
    print(pred)

    # epochs = 100
    # for epoch in range(epochs):
    # n.zero_grad()
    preds = [n(x) for x in inputs]
    loss = sum((ypred - yout) ** 2 for ypred, yout in zip(preds, ys))
    print(preds)

    # zero the gradients
    # forward pass
    # calculate loss(mse)
    # backprop
    # step of .01
