from nn.dataset import *
from nn.models import *
from nn.layers import *
from nn.optimizer import *
import matplotlib.pyplot as plt






if __name__ == "__main__":
    x, t = load_data()
    h = 0.01
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]
    scores = []
    losses = []

    _x, _t = load_data()
    _model = NNModel()
    _model.add_layer(Affine(2, 10))
    _model.add_layer(Sigmoid())
    _model.add_layer(Affine(10, 3))
    _model.add_loss_layer(SoftmaxWithLoss())
    _optimizer = SGD(lr=1)

    _total_loss = 0
    _loss_count = 0
    _loss_list = []
    _max_iters = len(_x) // 30
    for _epoch in range(500):
        _idx = np.random.permutation(len(_x))
        _x = _x[_idx]
        _t = _t[_idx]

        for _iters in range(_max_iters):
            _batch_x = _x[_iters*30:(_iters+1)*30]
            _batch_t = _t[_iters * 30:(_iters + 1) * 30]

            _loss = _model.forward(_batch_x, _batch_t)
            _model.backward()
            _optimizer.update(_model.params, _model.grads)

            _total_loss += _loss
            _loss_count += 1

            if (_iters + 1) % 50 == 0:
                avg_loss = _total_loss / _loss_count
                print('| epoch %d |  iter %d / %d | loss %.4f' % (_epoch + 1, _iters + 1, _max_iters, avg_loss))
                _loss_list.append(avg_loss)
                _total_loss, _loss_count = 0, 0

        if _epoch % 1 == 0:
            score = _model.predict(X)
            scores.append(score)

        if _epoch % 10 == 0:
            plt.plot(np.arange(len(_loss_list)), _loss_list, label='train')
            plt.xlabel('iterations (x10)')
            plt.ylabel('loss')
            plt.show(block=False)
            plt.pause(0.01)
    plt.show()

    Zs = []
    for score in scores:
        predict_cls = np.argmax(score, axis=1)
        Z = predict_cls.reshape(xx.shape)
        Zs.append(Z)

    j = 0
    for j in range(0, len(Zs)):
        plt.contourf(xx, yy, Zs[j])
        plt.axis('off')

        N = 500
        CLS_NUM = 3
        markers = ['o', 'x', '^']
        colors = ["r", "g", "b"]
        for i in range(CLS_NUM):
            plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i], color=colors[i])
        plt.show(block=False)
        plt.pause(0.1)
        print(j, len(Zs))
    plt.show()






















