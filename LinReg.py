import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
Y2 = []
fig = plt.figure
ax1 = plt.scatter(X, Y)
ax2 = plt.plot(X, Y2)

class LinReg:
    def __init__(self):
        self.points = []
        self.a = 0
        self.b = 0
        self.learning_rate_a = 0.0001
        self.learning_rate_b = 0.001
        self.nb_iter = 10000
        self.length = 0

    def set_points(self, x, y):
        self.points = [[i, j] for i, j in zip(x, y)]

    def calc(self):
        plt.show()
        self.length = len(self.points)
        for i in range(self.nb_iter):
            self.grad_descent()
        print(self.a, self.b)

    def grad_descent(self):
        global Y2
        grad_a = 0
        grad_b = 0
        Y2 = list(map(self.predict, (range(self.length))))
        plt.plot(X, Y2)
        plt.draw()
        for point in self.points:
            x = point[0]
            y = point[1]
            grad_a += - (2/self.length) * x * (y - ((self.a * x) + self.b))
            grad_b += - (2/self.length) * (y - ((self.a * x) + self.b))
        self.a -= (self.learning_rate_a * grad_a)
        self.b -= (self.learning_rate_b * grad_b)

    def predict(self, x):
        return (self.a * x + self.b)
