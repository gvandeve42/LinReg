from sklearn import datasets
from numpy import *
import matplotlib

def compute_error(b, m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, initial_b, initial_a, learning_rate, num_iterations):
    b = initial_b
    m = initial_a

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = [[1, 1], [2, 2], [3, 3], [4, 4]]#genfromtxt('world.csv', delimiter=',')
    # hyperparameters
    learning_rate = 0.0001
    # y = ax + b
    initial_a = 0
    initial_b = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_a, learning_rate, num_iterations)
    print(b)
    print(m)

if __name__ == '__main__':
    run()
