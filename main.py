import numpy as np
import matplotlib.pyplot as plt


def p(x):
    return np.cosh(x)


def q(x):
    return np.sinh(x)


def f(x):
    return np.cosh(x) + x * np.sinh(x)


def solution(x):
    return np.exp(-np.sinh(x)) + x


def solve(p, q, f, alpha, beta, gamma, x0, xend, h):
    xrange = np.arange(x0, xend, h)
    a = np.zeros(len(xrange))
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        a[i] = 1/(h**2) - p(x)/(2*h)
    a[-1] = - beta[1] / h

    b = np.zeros(len(xrange))
    b[0] = alpha[0] - beta[0] / h
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        b[i] = -2/(h**2) + q(x)
    b[-1] = alpha[1] + beta[1] / h

    c = np.zeros(len(xrange))
    c[0] = beta[0] / h
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        c[i] = 1/(h**2) + p(x)/(2*h)

    A = np.zeros(len(xrange))
    B = np.zeros(len(xrange))
    A[0] = -c[0]/b[0]
    B[0] = gamma[0]/b[0]
    for i, x in zip(range(1, len(xrange)), xrange[1:-1]):
        A[i] = -c[i] / (b[i] + a[i] * A[i-1])
        B[i] = (f(x) - a[i] * B[i-1]) / (b[i] + a[i] * A[i-1])
    B[-1] = (gamma[1] - a[-1] * B[-2]) / (b[-1] + a[-1] * A[-2])

    y = np.zeros(len(xrange))
    y[-1] = B[-1]
    for i in range(len(xrange) - 2, -1, -1):
        y[i] = B[i] + A[i] * y[i + 1]
    return y


def main1():
    x0 = 0
    xend = 1

    alpha = np.array([[0], [6]])
    beta = np.array([[1], [1]])
    gamma = np.array([[0], [8.3761]])

    # u'(0) = 0
    # 6u(1) + u'(1) = 8.3761

    sol = solution

    h = 0.01
    xrange = np.arange(x0, xend, h)

    y = solve(p, q, f, alpha, beta, gamma, x0, xend, h)

    number_of_methods = 1

    plt.subplot(2, number_of_methods, 1)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, y, color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.show()

if __name__ == '__main__':
    main1()
