import numpy as np
import matplotlib.pyplot as plt


m = 50
X = np.random.randn(m, 2)
y = np.int8(2.3 * X[:, 0] + 1.9 - X[:, 1] > 0)
c = ["r", "b"]

idx1 = np.where(y == 0)
idx2 = np.where(y == 1)

plt.scatter(X[idx1, 0], X[idx1, 1], marker="o", c="r")
plt.scatter(X[idx2, 0], X[idx2, 1], marker="o", c="b")
plt.show()


theta = [0, 0]
alpha = 0.5


def g(x):
    return 1.0 / (1.0 + np.e ** -x)


def h(x):
    global theta
    return g(theta[0] + theta[1] * x)


def J():
    global theta, X, y, m
    cost = 0
    for i in range(m):
        x = X[i][0]
        cost += y[i] * np.log(h(x)) + (1 - y[i]) * np.log(1 - h(x))
    return -cost / m


def dJ1():
    global theta, X, y, m
    result = 0
    for i in range(m):
        x = X[i][0]
        result += (h(x) - y[i]) * x
    return result / m


def dJ0():
    global theta, X, y, m
    result = 0
    for i in range(m):
        x = X[i][0]
        result += (h(x) - y[i])
    return result / m


def GradientDescent():
    global theta, alpha
    theta[0] -= alpha * dJ0()
    theta[1] -= alpha * dJ1()


for iter in range(500):
    print("Iteration:", iter, "Cost:", J(), "theta:", theta)
    GradientDescent()

    plt.cla()
    idx1 = np.where(y == 0)
    idx2 = np.where(y == 1)
    plt.scatter(X[idx1, 0], X[idx1, 1], marker="o", c="r")
    plt.scatter(X[idx2, 0], X[idx2, 1], marker="o", c="b")
    xx = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
    yy = xx * theta[1] + theta[0]
    plt.plot(xx, yy)
    plt.show(block=False)
    plt.pause(0.01)
plt.show()