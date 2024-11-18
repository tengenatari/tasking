import math

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


class NewtonInt:
    def __init__(self, func, n, a, b):
        self.table = np.zeros(shape=(n + 1, n + 1))
        x = np.linspace(a, b, n + 1)
        self.table[0] = func(x)
        self.n = n
        self.x0 = a
        self.h = x[1] - x[0]
        for i in range(1, n + 1):
            for j in range(1, n - i + 2):
                self.table[i][j - 1] = self.table[i - 1][j] - self.table[i - 1][j - 1]
        print(self.table)
    def calculate_points(self, points):
        ans = []
        for k in range(len(points)):
            sum = 0
            q = (points[k] - self.x0)/self.h
            for i in range(self.n + 1):
                p = 1

                for j in range(i):
                    p *= (q - j)
                sum += p * self.table[i][0]/ math.factorial(i)
            ans.append(sum)
        return ans





def lagranzh(x, y, points):

    ans = np.array([])
    for k in range(len(points)):
        sum = 0
        for i in range(len(x)):
            p = 1
            for j in range(len(y)):
                if j != i:
                    p *= (points[k] - x[j])/(x[i] - x[j])
            sum += p*y[i]
        ans = np.append(ans, sum)
    return ans
colors = plt.cm.jet(np.linspace(0,1,10))


def P(x, point):
    p = 1
    for i in range(len(x)):
        p *= point - x[i]
    return abs(p)


def R(x, point, f_b):
    return abs(f_b/math.factorial(len(x)) * P(x, point))
def R_max(x, points, f_b):
    max_ = []
    for i in range(len(points)):
        max_.append(R(x, points[i], f_b))
    return max(max_)


def main():




    x = np.linspace(0, 10, 150)
    y = np.exp(x)

    plt.figure(figsize=(8, 6), dpi=80)

    plt.subplot(3, 1, 1)
    plt.title('Лагранж')
    plt.title('Отрезок [0, 10], h = $\\frac{10}{20}$')
    plt.xlabel('x')
    plt.ylabel('$\\exp(x)$')

    x_ = np.linspace(0, 10, 151)
    plt.plot(x, y, color=colors[1], label='$\\exp(x)$')
    plt.plot(x_, lagranzh(x, y, x_), color=colors[5], label='$L_{' + str(len(x) - 1) + '}(x)$')
    plt.legend()



    plt.subplot(3, 1, 2)

    plt.title('Отрезок [1, 9], h = $\\frac{8}{150}$')
    plt.xlabel('x')
    plt.ylabel('$\\exp(x)$')

    x_ = np.linspace(1, 9, 151)
    plt.plot(x, y, color=colors[1], label='$\\exp(x)$')
    plt.plot(x_, lagranzh(x, y, x_), color=colors[5], label='$L_{' + str(len(x) - 1) + '}(x)$')
    plt.legend()

    plt.subplot(3, 2, 5)

    plt.title('Отрезок [9, 10], h = $\\frac{1}{150}$')
    plt.xlabel('x')
    plt.ylabel('$\\exp(x)$')

    x_ = np.linspace(9, 10, 151)
    plt.plot(x_, lagranzh(x, y, x_), color=colors[3], label='$L_{' + str(len(x) - 1) + '}(x)$')
    plt.legend()

    plt.subplot(3, 2, 6)

    plt.title('Отрезок [0, 1], h = $\\frac{1}{150}$')
    plt.xlabel('x')
    plt.ylabel('$\\exp(x)$')

    x_ = np.linspace(0, 1, 151)
    plt.plot(x_, lagranzh(x, y, x_), color=colors[4], label='$L_{' + str(len(x) - 1) + '}(x)$')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print('R(0.1) = ', R(x, 0.1, np.exp(10)))
    print("R(x') max = ", R_max(x, x_, np.exp(10)))
    print('R(0) = ', R(x, 0, np.exp(10)))



def newton():
    inter = NewtonInt(np.exp, 10, 0, 5)
    x = np.linspace(0, 10, 3)
    x_ = np.linspace(0, 10, 100)
    y = inter.calculate_points(x_)
    plt.plot(x_, y, color=colors[9], label='$L_{Newton}(x)$')
    plt.plot(x_, np.exp(x_), color=colors[7], label='$L_{Newton}(x)$')
    plt.plot(x_,lagranzh(x, np.exp(x), x_), color=colors[5], label='$L_{' + str(len(x) - 1) + '}(x)$')
    plt.xlabel('x')
    plt.ylabel('$\\exp(x)$')
    plt.legend()
    plt.show()

main()