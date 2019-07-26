class NNModel:
    def __init__(self):
        self.layers = []
        self.params, self.grads = [], []
        self.loss_layer = None

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.grads += layer.grads

    def add_loss_layer(self, loss_layer):
        self.loss_layer = loss_layer

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
