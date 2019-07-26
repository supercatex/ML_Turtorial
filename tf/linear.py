import numpy as np
import matplotlib.pyplot as plt

# y = kx + b, k = 2.3, b = 1.9
m = 50
X = np.random.randn(m, 1)
y = 2.3 * X + 1.9 + np.random.randn(m, 1) * 1

plt.scatter(X, y)
plt.show()


theta = [0, 0]
alpha = 0.03


def h(x):
    global theta
    return theta[0] + theta[1] * x


def J():
    global theta, X, y, m
    cost = 0
    for i in range(m):
        x = X[i][0]
        cost += (h(x) - y[i][0]) ** 2
    return cost / 2 / m


def dJ1():
    global theta, X, y, m
    result = 0
    for i in range(m):
        x = X[i][0]
        result += (h(x) - y[i][0]) * x
    return result / m


def dJ0():
    global theta, X, y, m
    result = 0
    for i in range(m):
        x = X[i][0]
        result += (h(x) - y[i][0])
    return result / m


def GradientDescent():
    global theta, alpha
    theta[0] -= alpha * dJ0()
    theta[1] -= alpha * dJ1()


for iter in range(100):
    print("Iteration:", iter, "Cost:", J(), "theta:", theta)
    GradientDescent()

    plt.cla()
    plt.scatter(X, y)
    xx = np.linspace(np.min(X), np.max(X))
    yy = xx * theta[1] + theta[0]
    plt.plot(xx, yy)
    plt.show(block=False)
    plt.pause(0.01)
plt.show()