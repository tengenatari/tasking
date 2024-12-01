from matplotlib import pyplot as plt
import numpy as np
from gauss import gauss_solution




class LinearSplainInterpolation:

    def __init__(self):
        self.points = [0, 1]
        self.func = [0, 1]
        self.otr = [[1, 0]]

    def build(self, points, func):
        self.points = list(points)
        self.func = list(func)
        self.otr = []
        for i in range(1, len(points)):
            self.otr.append(self.step(i))


    def step(self, i):
        y = self.func
        x = self.points
        delta_y = y[i] - (y[i - 1]*x[i])/x[i - 1]
        delta_x = 1 - x[i]/x[i - 1]
        b = delta_y/delta_x
        a = (y[i] - b)/x[i]
        return [a, b]

    def get_points(self, points):
        points.sort()
        y = []
        for x in points:
            i = self.points.index(x)
            y.append(self.otr[i][0]*x + self.otr[i][1])
        return y

class TrioInterpolation:
    def __init__(self):
        self.matrix = None
        self.points = None
        self.func = None
        self.step = 0
        self.a = None
        self.b = None
        self.c = None
        self.d = None
    def build(self, x, y):
        self.step = x[1] - x[0]
        self.points = list(x)
        self.matrix = [[0]*(len(x) + 1) for _ in range(len(x))]

        #вычисление c
        self.matrix[0][0] = self.matrix[-1][-3] = self.step
        self.matrix[0][1] = self.matrix[-1][-2] = 4*self.step
        self.matrix[0][-1] = self.matrix[-1][-1] = 0
        for i in range(1, len(x) - 1):
            self.matrix[i][i - 1] = self.matrix[i][i + 1] = self.step
            self.matrix[i][i] = 4*self.step
            self.matrix[i][-1] = 3*(y[i - 1] - 2*y[i] +y[i + 1])/self.step
        for _ in range(len(self.matrix)):
            print(*self.matrix[_])
        self.c, _, _ =gauss_solution(list(self.matrix))

        #вычисление d

        self.d = []

        for i in range(len(self.c) - 1):
            self.d.append((self.c[i + 1] - self.c[i])/(3*self.step))

        #вычисление b

        self.b = []

        for i in range(1, len(self.c) - 2):
            delta_y = (y[i] - y[i - 1])/self.step
            delta_c = - self.step/3*(self.c[i + 2] + 2*self.c[i + 1])
            self.b.append(delta_y - delta_c)

        #вычисление a
        self.a = []
        for i in range(1, len(x)):
            self.a.append(y[i - 1])

    def get_points(self, points):
        points.sort()
        y = []
        for x in points:
            i = self.points.index(x)
            y.append(self.a[i] + self.b[i]*x + self.c[i]*x**2 + self.d[i]*x**3)
        return y

trio = TrioInterpolation()
trio.build([1.1, 2.2 ,3.3 ,4.4,5.5 ,6.6], [1.1, 2.5,3.9,4.1,5.1,6])







x_0 = np.arange(0, 10, 0.1)

sin_x_0 = np.sin(x_0)

lin = LinearSplainInterpolation()
tim = TrioInterpolation()

tim.build(x_0, sin_x_0)

lin.build(x_0, sin_x_0)

plt.figure(figsize=(8, 6), dpi=80)

plt.subplot(3, 1, 1)
print(x_0)
print(sin_x_0)
plt.plot(x_0, sin_x_0, color="red")
plt.plot(np.arange(0, 10, 1), lin.get_points(np.arange(0, 10, 1)), color="orange")

plt.subplot(3, 1, 2)
print(x_0)
print(sin_x_0)
plt.plot(x_0, sin_x_0)
plt.plot(np.arange(0, 10, 1), tim.get_points(np.arange(1, 10, 0.1)), color="orange")
plt.tight_layout()
plt.show()
print(lin.get_points(np.arange(0, 10, 1)))
