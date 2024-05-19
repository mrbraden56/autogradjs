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

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):  # -self
        return self * -1

    def __rsub__(self, other):  # other - self
        return other + (-self)

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
            y = sum(((wi * xi).relu() for wi, xi in zip(neuron, inputs)), b)
            out.append(y)
        return out


class FFN:
    def __init__(self):
        self.layers = [Layer(3, 16), Layer(16, 8), Layer(8, 1)]

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


if __name__ == "__main__":
    inputs = [
        [random.uniform(-1, 1) for _ in range(10)],
        [random.uniform(-1, 1) for _ in range(10)],
        [random.uniform(-1, 1) for _ in range(10)],
        [random.uniform(-1, 1) for _ in range(10)],
        [random.uniform(-1, 1) for _ in range(10)],
        [random.uniform(-1, 1) for _ in range(10)],
    ]
    ys = [
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1),
    ]
    n = FFN()
    pred = [n(x) for x in inputs]

    epochs = 500
    for epoch in range(epochs):
        n.zero_grad()
        preds = [n(x) for x in inputs]
        loss = sum(((ypred - yout) ** 2 for ypred, yout in zip(preds, ys)), Tensor(0))
        loss.backward()
        n.sgd(0.001)
        print(f"Epoch: {epoch}, Loss: {loss.value}")

    print(preds)
