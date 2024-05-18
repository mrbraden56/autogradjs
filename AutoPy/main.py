class tensor:
    def __init__(self, value, parents=None, grad=None):
        self.value = value
        self.parents = parents
        self._backward = lambda: None
        self.grad = 1

    def __repr__(self):
        return f"Tensor <{self.value}>"

    def __add__(self, x2):
        if isinstance(x2, tensor):
            y = tensor(self.value + x2.value, (self, x2))
        else:
            x2 = tensor(x2)
            y = tensor(self.value + x2.value, (self, x2))

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
        if isinstance(x2, tensor):
            y = tensor(self.value * x2.value, (self, x2))
        else:
            x2 = tensor(x2)
            y = tensor(self.value * x2.value, (self, x2))

        def backward():
            self.grad += x2.value * y.grad
            x2.grad += self.value * y.grad

        y._backward = backward
        return y

    def backward(self):
        # NOTE: Sorting topologically is necessary so that we can call the backwards pass in order. If we do it out of order we would be
        # mulyiplying the wrong things for the chain rule
        pass


if __name__ == "__main__":
    a = tensor(7)
    b = tensor(8)
