class tensor:
    def __init__(self, value, parents=None):
        self.value = value
        self.parents = parents

    def __repr__(self):
        return f"Tensor <{self.value}>"

    def __add__(self, x2):
        # NOTE: Gradient = 1.0
        if isinstance(x2, tensor):
            return tensor(self.value + x2.value, (self, x2))
        else:
            ten_x2 = tensor(x2)
            return tensor(self.value + ten_x2.value, (self, ten_x2))

    def __mul__(self, x2):
        # NOTE: Gradient = x2
        if isinstance(x2, tensor):
            return tensor(self.value * x2.value, (self, x2))
        else:
            ten_x2 = tensor(x2)
            return tensor(self.value * ten_x2.value, (self, ten_x2))


if __name__ == "__main__":
    a = tensor(7)
    b = tensor(8)
