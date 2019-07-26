
class NeuralNetworkModel(object):
    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []
        self.loss = None
        self.optimizer = None
        self.loss_layer = Softmax()

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_optimizer(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss = loss_function
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        y = self.predict(x)
        self.loss_layer.forward(y)
        return self.loss(y, t)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


if __name__ == "__main__":
    from common.layers import Affine
    from common.activation import Sigmoid, Softmax
    from common.loss import cross_entropy_error
    from common.optimizer import SGD
    import numpy as np
    from common.dataset import load_data


    _model = NeuralNetworkModel()
    _model.add_layer(Affine(2, 4))
    _model.add_layer(Sigmoid())
    _model.add_layer(Affine(4, 3))
    # _model.add_layer(Softmax())
    _model.add_optimizer(SGD(), cross_entropy_error)

    _x, _t = load_data(100, 3)

    for i in range(5):
        _loss = _model.forward(_x, _t)
        print(np.sum(_loss))
        _model.backward()
        _model.optimizer.update(_model.params, _model.grads)
