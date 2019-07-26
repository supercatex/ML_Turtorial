import numpy as np


def load_data(N=500, CLS_NUM=3):
    DIM = 2

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 100
    CLS_NUM = 3
    x, t = load_data(N, CLS_NUM)

    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
    plt.show()
