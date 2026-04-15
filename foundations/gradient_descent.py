class Solution:
    def get_minimizer(self, iterations: int, lr: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        token=init
        for _ in range(iterations):
            deriv=2*token
            token=token-lr*deriv
        return round(token,5)
