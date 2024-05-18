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
            # NOTE:
            # y = a + b
            # dy / da = 1 * (previous chained method)
            # dy / db = 1 * (previous chained method)
            self.grad += 1.0 * (y.grad)
            x2.grad += 1.0 * (y.grad)

        y._backward = backward
        return y

    def __mul__(self, x2):
        # NOTE: Gradient = x2
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


class Linear:
    def __init__(self, nin, nout):
        w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        b = Tensor(0)


class FFN:
    def __init__(self):
        nin = 32
        nout = 256
        self.l1 = Linear(nin, nout)
        self.l2 = Linear(256, 32)
        self.l3 = Linear(32, 1)

    def forward(self, x):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        return f3


if __name__ == "__main__":
    x = [[Tensor(random.uniform(-1, 1)) for _ in range(32)] for _ in range(32)]
    print(f"Shape of input is ({len(x)}, {len(x[0])})")

    epochs = 100
    for epoch in range(epochs):
        pass

        # zero the gradients
        # forward pass
        # calculate loss(mse)
        # backprop
        # step of .01
