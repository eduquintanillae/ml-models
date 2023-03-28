from utils.linear_algebra import Vector, dot, distance, add, scalar_multiply, vector_mean
from typing import TypeVar, List, Iterator, Callable
import random
import matplotlib.pyplot as plt

T = TypeVar('T')  # "generic" functions

class GradientDescent:

    def sum_of_squares(self, v: Vector) -> float:
        """Computes the sum of squared elements in v"""
        return dot(v, v)

    def difference_quotient(self, f: Callable[[float], float],
                            x: float,
                            h: float) -> float:
        return (f(x + h) - f(x)) / h

    def square(self, x: float) -> float:
        return x * x

    def derivative(self, x: float) -> float:
        return 2 * x
    
    def partial_difference_quotient(self, f: Callable[[Vector], float],
                                    v: Vector,
                                    i: int,
                                    h: float) -> float:
        """Returns the i-th partial difference quotient of f at v"""
        w = [v_j + (h if j == i else 0)
             for j, v_j in enumerate(v)]
    
        return (f(w) - f(v)) / h

    def estimate_gradient(self, f: Callable[[Vector], float],
                        v: Vector,
                        h: float = 0.0001):
        return [self.partial_difference_quotient(f, v, i, h)
                for i in range(len(v))]


    def gradient_step(self, v: Vector, gradient: Vector, step_size: float) -> Vector:
        """Moves `step_size` in the `gradient` direction from `v`"""
        assert len(v) == len(gradient)
        step = scalar_multiply(step_size, gradient)
        return add(v, step)

    def sum_of_squares_gradient(self, v: Vector) -> Vector:
        return [2 * v_i for v_i in v]

    def linear_gradient(self, x: float, y: float, theta: Vector) -> Vector:
        slope, intercept = theta
        predicted = slope * x + intercept
        error = (predicted - y)
        squared_error = error ** 2
        grad = [2 * error * x, 2 * error]
        return grad

    def minibatches(self, dataset: List[T],
                    batch_size: int,
                    shuffle: bool = True) -> Iterator[List[T]]:
        """Generates `batch_size`-sized minibatches from the dataset"""
        batch_starts = [start for start in range(0, len(dataset), batch_size)]

        if shuffle: random.shuffle(batch_starts)

        for start in batch_starts:
            end = start + batch_size
            yield dataset[start:end]

def main():
    gd = GradientDescent()
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
    
    # Generate some linear data with noise
    xs = range(-10, 11)
    actuals = [gd.derivative(x) for x in xs]
    estimates = [gd.difference_quotient(gd.square, x, h=0.001) for x in xs]
    
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, 'rx', label='Actual')
    plt.plot(xs, estimates, 'b+', label='Estimate')
    plt.legend(loc=9)
    plt.close()
    
    
    # Using Gradient Descent to Minimize Functions
    v = [random.uniform(-10, 10) for i in range(3)]
    
    for epoch in range(1000):
        grad = gd.sum_of_squares_gradient(v)    # compute the gradient at v
        v = gd.gradient_step(v, grad, -0.01)    # take a negative gradient step
        print(epoch, v)
    
    assert distance(v, [0, 0, 0]) < 0.001    # v should be close to 0
    
    
    # Using Gradient Descent to Fit Models
    # Start with random values for slope and intercept.
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.001
    
    for epoch in range(5000):
        grad = vector_mean([gd.linear_gradient(x, y, theta) for x, y in inputs]) # Mean of the gradients
        theta = gd.gradient_step(theta, grad, -learning_rate) # Take a step in that direction
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
    
    # Minibatch gradient descent example
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(1000):
        for batch in gd.minibatches(inputs, batch_size=20):
            grad = vector_mean([gd.linear_gradient(x, y, theta) for x, y in batch])
            theta = gd.gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
    
    # Stochastic gradient descent example
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    for epoch in range(100):
        for x, y in inputs:
            grad = gd.linear_gradient(x, y, theta)
            theta = gd.gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
if __name__ == "__main__": main()