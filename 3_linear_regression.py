from typing import Tuple
import random
import tqdm
from utils.linear_algebra import Vector
from utils.statistics import (correlation, standard_deviation, 
                              mean, num_friends_good, daily_minutes_good, 
                              de_mean)
import importlib
gradient_descent_module = importlib.import_module('0_gradient_descent')
gradient_step = gradient_descent_module.GradientDescent().gradient_step

class LinearRegression:
    def predict(self, alpha: float, beta: float, x_i: float) -> float:
        return beta * x_i + alpha

    def error(self, alpha: float, beta: float, x_i: float, y_i: float) -> float:
        """
        The error from predicting beta * x_i + alpha
        when the actual value is y_i
        """
        return self.predict(alpha, beta, x_i) - y_i


    def sum_of_sqerrors(self, alpha: float, beta: float, x: Vector, y: Vector) -> float:
        return sum(self.error(alpha, beta, x_i, y_i) ** 2
                for x_i, y_i in zip(x, y))


    def least_squares_fit(self, x: Vector, y: Vector) -> Tuple[float, float]:
        """
        Given two vectors x and y,
        find the least-squares values of alpha and beta
        """
        beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
        alpha = mean(y) - beta * mean(x)
        return alpha, beta


    def total_sum_of_squares(self, y: Vector) -> float:
        """the total squared variation of y_i's from their mean"""
        return sum(v ** 2 for v in de_mean(y))

    def r_squared(self, alpha: float, beta: float, x: Vector, y: Vector) -> float:
        """
        the fraction of variation in y captured by the model, which equals
        1 - the fraction of variation in y not captured by the model
        """
        return 1.0 - (self.sum_of_sqerrors(alpha, beta, x, y) /
                    self.total_sum_of_squares(y))

def main():
    num_epochs = 10000
    random.seed(0)
    guess = [random.random(), random.random()]  # choose random value to start
    learning_rate = 0.00001
    linear_regression = LinearRegression()
    
    with tqdm.trange(num_epochs) as t:
        for _ in t:
            alpha, beta = guess
    
            # Partial derivative of loss with respect to alpha
            grad_a = sum(2 * linear_regression.error(alpha, beta, x_i, y_i)
                         for x_i, y_i in zip(num_friends_good,
                                             daily_minutes_good))
    
            # Partial derivative of loss with respect to beta
            grad_b = sum(2 * linear_regression.error(alpha, beta, x_i, y_i) * x_i
                         for x_i, y_i in zip(num_friends_good,
                                             daily_minutes_good))
    
            # Compute loss to stick in the tqdm description
            loss = linear_regression.sum_of_sqerrors(alpha, beta,
                                   num_friends_good, daily_minutes_good)
            t.set_description(f"loss: {loss:.3f}")
    
            # Finally, update the guess
            guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
    
    alpha, beta = guess
    print(f"alpha: {alpha}, beta: {beta}")
    
if __name__ == "__main__": 
    main()