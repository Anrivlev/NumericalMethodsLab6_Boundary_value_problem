import numpy as np
import matplotlib.pyplot as plt

# y'' + p(x)y' + q(x)y = f(x), 0 < x < l
# alpha1 * y(0) + beta1 * y'(0) = gamma1
# alpha1 * y(l) + beta1 * y'(l) = gamma1


def p(x):
    return np.cosh(x)


def q(x):
    return np.sinh(x)


def f(x):
    return np.cosh(x) + x * np.sinh(x)


def solution(x):
    return np.exp(-np.sinh(x)) + x


def least_squares(x, y):
    n = len(x)

    sumx = x.sum()
    sumy = y.sum()
    xy = x * y
    sumxy = xy.sum()
    xx = x * x
    sumxx = xx.sum()

    b = (n * sumxy - sumx*sumy) / (n * sumxx - sumx**2)
    a = (sumy - b * sumx) / n
    return a, b


def solve1(p, q, f, alpha, beta, gamma, x0, xend, h):
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

    f_array = np.zeros(len(xrange))
    f_array[0] = gamma[0]
    f_array[-1] = gamma[1]
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        f_array[i] = f(x)

    return solve3diagonal(a, b, c, f_array)


def solve2(p, q, f, alpha, beta, gamma, x0, xend, h):
    xrange = np.arange(x0, xend, h)

    a = np.zeros(len(xrange))
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        a[i] = 1 / (h ** 2) - p(x) / (2 * h)
    a[-1] = 2

    b = np.zeros(len(xrange))
    b[0] = q(x0) * h**2 - 2 + (2 * h * alpha[0]) / beta[0] - (alpha[0] * p(x0) * h**2) / beta[0]
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        b[i] = -2 / (h ** 2) + q(x)
    b[-1] = q(xend) * h**2 - 2 - (2 * h * alpha[1]) / beta[1] - (alpha[1] * p(xend) * h**2) / beta[1]

    c = np.zeros(len(xrange))
    c[0] = 2
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        c[i] = 1 / (h ** 2) + p(x) / (2 * h)

    f_array = np.zeros(len(xrange))
    f_array[0] = f(x0) * h**2 + (2 * h * gamma[0]) / beta[0] - (p(x0) * gamma[0] * h**2) / beta[0]
    f_array[-1] = f(xend) * h**2 - (2 * h * gamma[1]) / beta[1] - (p(xend) * gamma[1] * h**2) / beta[1]
    for i, x in zip(range(1, len(xrange) - 1), xrange[1:-1]):
        f_array[i] = f(x)

    return solve3diagonal(a, b, c, f_array)


def solve3diagonal(a, b, c, f):
    # assuming that len(a) == len(b) == len(c) == len(f) == n + 1
    A = np.zeros(len(f))
    B = np.zeros(len(f))
    A[0] = -c[0] / b[0]
    B[0] = f[0] / b[0]
    for i in range(1, len(f) - 1):
        A[i] = -c[i] / (b[i] + a[i] * A[i - 1])
        B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1])
    A[-1] = 0
    B[-1] = (f[-1] - a[-1] * B[-2]) / (b[-1] + a[-1] * A[-2])

    y = np.zeros(len(f))
    y[-1] = B[-1]
    for i in range(len(f) - 2, -1, -1):
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

    h = 0.001
    xrange = np.arange(x0, xend, h)

    y = solve2(p, q, f, alpha, beta, gamma, x0, xend, h)

    number_of_methods = 1

    plt.subplot(2, number_of_methods, 1)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()
    plt.plot(xrange, y, color='k', label='Численное значение')
    plt.plot(xrange, sol(xrange), ls='--', color='k', label='Аналитическое значение')

    plt.show()


def main2():
    x0 = 0
    xend = 1

    alpha = np.array([[0], [6]])
    beta = np.array([[1], [1]])
    gamma = np.array([[0], [8.3761]])

    sol = solution

    hmin = 0.0001
    hmax = 0.1
    hstep = 0.0001
    hrange = np.arange(hmin, hmax, hstep)

    error = dict()
    error[solve1] = np.zeros(len(hrange))
    error[solve2] = np.zeros(len(hrange))

    for i, h in zip(range(len(hrange)), hrange):
        for key in error:
            xrange = np.arange(x0, xend, h)
            error[key][i] = np.max(np.abs(key(p, q, f, alpha, beta, gamma, x0, xend, h) - sol(xrange)))

    hrange = np.log10(hrange)
    for key in error:
        error[key] = np.log10(error[key])

    plt.suptitle('Зависимость логарифма абсолютной погрешности от логарифма шага интегрирования')
    for key, i in zip(error, range(1, len(error) + 1)):
        plt.subplot(1, len(error), i)
        plt.title(key.__name__)
        plt.xlabel("log(h)")
        plt.ylabel("log(max(|Δu|))")
        plt.grid()
        plt.plot(hrange, error[key], color='k')

    for key in error:
        coeffs = least_squares(hrange, error[key])
        print(key.__name__, ": ", coeffs[0], " + ", coeffs[1], "x", sep="")
    plt.show()


if __name__ == '__main__':
    main2()
