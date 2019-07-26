from common.net import ForwardBackwardNet
import numpy as np


class Sigmoid(ForwardBackwardNet):
    def __init__(self):
        self.out = None
        self.params = []
        self.grads = []

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Softmax(ForwardBackwardNet):
    def __init__(self):
        self.out = None
        self.params = []
        self.grads = []

    def forward(self, x):
        e = np.exp(x)
        self.out = e / np.sum(e, axis=1).reshape(x.shape[0], 1)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


if __name__ == "__main__":
    from common.loss import cross_entropy_error

    _x = np.array([[1, 2, 3], [2, 3, 4]])
    _a = Softmax()

    _y = _a.forward(_x)
    print(_y)

    _t = np.array([[1, 0, 0], [0, 0, 1]])
    print(_t)

    print(cross_entropy_error(_y, _t))
