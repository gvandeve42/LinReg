class LinReg:
    def __init__(self, array):
        self.points = array
        self.a = 0
        self.b = 0
        self.learningrate = 0.00001
        self.nb_iter = 10000
        self.length = len(array)

    def calc(self):
        for i in range(self.nb_iter):
            self.grad_descent()
        print(self.a, self.b)

    def grad_descent(self):
        grad_a = 0
        grad_b = 0
        for point in self.points:
            x = point[0]
            y = point[1]
            grad_a += - (1/self.length) * x * (y - (self.a * x) + self.b)
            grad_b += - (1/self.length) * (y - (self.a * x) + self.b)
        self.a -= (self.learningrate * grad_a)
        self.b -= (self.learningrate * grad_b)

test = LinReg([[1, 1],
               [2, 2],
               [3, 3],
               [4, 4],
               [5, 5],
               [6, 6],
               [7, 7],
               [8, 8],
               [9, 9],
               [10, 10],
               [20, 20],
               [30, 30],
               [40, 40],
               [50, 50],
               [1000, 1000]])
test.calc()
