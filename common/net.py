import abc


class ForwardBackwardNet(abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass
