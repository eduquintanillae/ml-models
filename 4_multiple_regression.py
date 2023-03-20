from typing import List
import random
import tqdm
from utils.linear_algebra import dot, Vector
from utils.linear_algebra import vector_mean
from typing import TypeVar, Callable
from utils.statistics import median, standard_deviation
from utils.probability import normal_cdf
from utils.statistics import daily_minutes_good
from utils.linear_algebra import add
from utils.statistics import daily_minutes_good
from typing import Tuple

import importlib
gradient_descent_module = importlib.import_module('0_gradient_descent')
gradient_step = gradient_descent_module.GradientDescent().gradient_step
linear_regression_module = importlib.import_module('3_linear_regression')
total_sum_of_squares = linear_regression_module.LinearRegression().total_sum_of_squares


X = TypeVar('X')        # Generic type for data
Stat = TypeVar('Stat')  # Generic type for "statistic"

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]

class MultipleRegression:
    def predict(self, x: Vector, beta: Vector) -> float:
        """assumes that the first element of x is 1"""
        return dot(x, beta)

    def error(self, x: Vector, y: float, beta: Vector) -> float:
        return self.predict(x, beta) - y

    def squared_error(self, x: Vector, y: float, beta: Vector) -> float:
        return self.error(x, y, beta) ** 2

    def sqerror_gradient(self, x: Vector, y: float, beta: Vector) -> Vector:
        err = self.error(x, y, beta)
        return [2 * err * x_i for x_i in x]

    def least_squares_fit(self, xs: List[Vector],
                        ys: List[float],
                        learning_rate: float = 0.001,
                        num_steps: int = 1000,
                        batch_size: int = 1) -> Vector:
        """
        Find the beta that minimizes the sum of squared errors
        assuming the model y = dot(x, beta).
        """
        # Start with a random guess
        guess = [random.random() for _ in xs[0]]

        for _ in tqdm.trange(num_steps, desc="least squares fit"):
            for start in range(0, len(xs), batch_size):
                batch_xs = xs[start:start+batch_size]
                batch_ys = ys[start:start+batch_size]

                gradient = vector_mean([self.sqerror_gradient(x, y, guess)
                                        for x, y in zip(batch_xs, batch_ys)])
                guess = gradient_step(guess, gradient, -learning_rate)

        return guess

    def multiple_r_squared(self, xs: List[Vector], ys: Vector, beta: Vector) -> float:
        sum_of_squared_errors = sum(self.error(x, y, beta) ** 2
                                    for x, y in zip(xs, ys))
        return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)

    def bootstrap_sample(self, data: List[X]) -> List[X]:
        """randomly samples len(data) elements with replacement"""
        return [random.choice(data) for _ in data]

    def bootstrap_statistic(self, data: List[X],
                            stats_fn: Callable[[List[X]], Stat],
                            num_samples: int) -> List[Stat]:
        """evaluates stats_fn on num_samples bootstrap samples from data"""
        return [stats_fn(self.bootstrap_sample(data)) for _ in range(num_samples)]

    def p_value(self, beta_hat_j: float, sigma_hat_j: float) -> float:
        if beta_hat_j > 0:
            # if the coefficient is positive, we need to compute twice the
            # probability of seeing an even *larger* value
            return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
        else:
            # otherwise twice the probability of seeing a *smaller* value
            return 2 * normal_cdf(beta_hat_j / sigma_hat_j)

    def ridge_penalty(self, beta: Vector, alpha: float) -> float:
        return alpha * dot(beta[1:], beta[1:])

    def squared_error_ridge(self, x: Vector,
                            y: float,
                            beta: Vector,
                            alpha: float) -> float:
        """estimate error plus ridge penalty on beta"""
        return self.error(x, y, beta) ** 2 + self.ridge_penalty(beta, alpha)

    def ridge_penalty_gradient(self, beta: Vector, alpha: float) -> Vector:
        """gradient of just the ridge penalty"""
        return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

    def sqerror_ridge_gradient(self, x: Vector,
                            y: float,
                            beta: Vector,
                            alpha: float) -> Vector:
        """
        the gradient corresponding to the ith squared error term
        including the ridge penalty
        """
        return add(self.sqerror_gradient(x, y, beta),
                self.ridge_penalty_gradient(beta, alpha))

    def least_squares_fit_ridge(self, xs: List[Vector],
                                ys: List[float],
                                alpha: float,
                                learning_rate: float,
                                num_steps: int,
                                batch_size: int = 1) -> Vector:
        # Start guess with mean
        guess = [random.random() for _ in xs[0]]

        for i in range(num_steps):
            for start in range(0, len(xs), batch_size):
                batch_xs = xs[start:start+batch_size]
                batch_ys = ys[start:start+batch_size]

                gradient = vector_mean([self.sqerror_ridge_gradient(x, y, guess, alpha)
                                        for x, y in zip(batch_xs, batch_ys)])
                guess = gradient_step(guess, gradient, -learning_rate)

        return guess

    def lasso_penalty(self, beta, alpha):
        return alpha * sum(abs(beta_i) for beta_i in beta[1:])

def main():
    random.seed(0)
    learning_rate = 0.001
    multiple_regression = MultipleRegression()
    
    beta = multiple_regression.least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
    
    def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
        x_sample = [x for x, _ in pairs]
        y_sample = [y for _, y in pairs]
        beta = multiple_regression.least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
        print("bootstrap sample", beta)
        return beta

    bootstrap_betas = multiple_regression.bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                          estimate_sample_beta,
                                          100)
    
    bootstrap_standard_errors = [
        standard_deviation([beta[i] for beta in bootstrap_betas])
        for i in range(4)]
    
    print(bootstrap_standard_errors)
    
    beta_0 = multiple_regression.least_squares_fit_ridge(inputs, daily_minutes_good, 0.0,  # alpha
                                     learning_rate, 5000, 25)
    
    beta_0_1 = multiple_regression.least_squares_fit_ridge(inputs, daily_minutes_good, 0.1,  # alpha
                                       learning_rate, 5000, 25)
    
    beta_1 = multiple_regression.least_squares_fit_ridge(inputs, daily_minutes_good, 1,  # alpha
                                     learning_rate, 5000, 25)
    
    beta_10 = multiple_regression.least_squares_fit_ridge(inputs, daily_minutes_good,10,  # alpha
                                      learning_rate, 5000, 25)
    
    print("Ridge regression results:")
    print("alpha=0.0:", beta_0)
    print("alpha=0.1:", beta_0_1)
    print("alpha=1.0:", beta_1)
    print("alpha=10.0:", beta_10)
    
if __name__ == "__main__": 
    main()