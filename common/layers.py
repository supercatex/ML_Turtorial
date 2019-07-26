from common.net import ForwardBackwardNet
import numpy as np


class Affine(ForwardBackwardNet):
    def __init__(self, input_units=0, units=1):
        self.units = units
        self.W = np.random.randn(input_units, units)
        self.b = np.random.randn(units)
        self.x = None
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


if __name__ == "__main__":
    from common.activation import Sigmoid, Softmax

    _x = np.random.randn(10, 2)

    _model = [
        Affine(2, 4),
        Sigmoid(),
        Affine(4, 3),
        Softmax()
    ]

    _i = _x
    for _layer in _model:
        _i = _layer.forward(_i)
    print(_i)
