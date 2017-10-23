import csv
from math import *
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self):
        self.points = []
        self.a = 0
        self.b = 0
        self.learning_rate_a = 0.001
        self.learning_rate_b = 0.1
        self.nb_iter = 1000
        self.length = 0

    def set_points(self, x, y):
        self.points = [[i, j] for i, j in zip(x, y)]

    def calc(self):
        global ax1
        global ax2
        global Y2
        self.length = len(self.points)
        for i in range(self.nb_iter):
            self.grad_descent()
        print(self.a, self.b)

    def grad_descent(self):
        grad_a = 0
        grad_b = 0
        for point in self.points:
            x = point[0]
            y = point[1]
            grad_a += - (2/self.length) * x * (y - ((self.a * x) + self.b))
            grad_b += - (2/self.length) * (y - ((self.a * x) + self.b))
        self.a -= (self.learning_rate_a * grad_a)
        self.b -= (self.learning_rate_b * grad_b)

    def predict(self, x):
        return (self.a * x + self.b)

def init_data(csv_name, row_X, row_Y):
    X = []
    Y = []
    with open(csv_name, 'r') as csvfile:
        doc = csv.reader(csvfile, delimiter=',')
        next(doc, None)
        for row in doc:
            X.append(float(row[row_X]))
            Y.append(float(row[row_Y]))
        return X, Y

def run_test():
    X, Y = init_data('pov.csv', 1, 2)
    test = LinReg()
    test.set_points(X, Y)
    test.calc()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    Y2 = list(map(test.predict, (range(test.length))))
    ax1 = plt.scatter(X, Y)
    ax2 = plt.plot(range(test.length), Y2)
    plt.show()

if __name__ == '__main__':
    run_test()
