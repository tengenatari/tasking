from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
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
colors = plt.cm.jet(np.linspace(0,1,2))
def main():
    x = np.linspace(0, 10, 100)
    y = np.exp(x)
    print(y)
    x_ =np.linspace(0, 10, 150)
    y_ = lagranzh(x, y, x_)
    plt.plot(x, y, color=colors[0])
    plt.plot(x_, y_, color=colors[1])
    plt.show()
main()