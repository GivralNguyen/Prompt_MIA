import torch
from math import sqrt, exp
import numpy as np
from scipy.special import erf
class LaplaceMechanism:
    def __init__(self, epsilon, sensitivity=1.0):
        """
        Initializes the LaplaceMechanism with the given privacy budget (epsilon) and sensitivity.
        
        :param epsilon: Privacy budget (lower values add more noise).
        :param sensitivity: Sensitivity of the data (default is 1.0).
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive for differential privacy.")
        
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def add_noise(self, data):
        """
        Adds Laplace noise to input data for Local Differential Privacy (LDP).
        
        :param data: Input tensor or scalar value.
        :return: Noisy data tensor.
        """
        # Ensure data is a PyTorch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        noise = torch.distributions.Laplace(0, self.scale).sample(data.shape).to(data.device)
        return data + noise


class GaussianMechanism:
    def __init__(self, epsilon, delta, sensitivity=1.0):
        """
        Initializes the GaussianMechanism with the given privacy budget (epsilon), privacy relaxation (delta), and sensitivity.
        
        :param epsilon: Privacy budget (lower values add more noise).
        :param delta: Privacy relaxation parameter (small value, e.g., 1e-5).
        :param sensitivity: Sensitivity of the data (default is 1.0).
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("Epsilon must be positive, and 0 < delta < 1.")
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Compute the noise standard deviation
        self.sigma = (sensitivity / epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / delta)))
    
    def add_noise(self, data):
        """
        Adds Gaussian noise to the input data for (ε, δ)-Differential Privacy.
        
        :param data: Input tensor or scalar.
        :return: Noisy data tensor.
        """
        # Ensure data is a PyTorch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        print(self.sigma)
        noise = torch.normal(mean=0, std=self.sigma, size=data.shape).to(data.device)
        return data + noise

class AnalyticGaussianMechanism:
    def __init__(self, epsilon, delta, GS, tol=1e-12):
        """
        Initializes the AnalyticGaussianMechanism with the given privacy parameters and global sensitivity.

        :param epsilon: Privacy budget (epsilon > 0)
        :param delta: Privacy relaxation parameter (0 < delta < 1)
        :param GS: Upper bound on L2 global sensitivity (GS >= 0)
        :param tol: Error tolerance for binary search (default is 1.e-12)
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("Epsilon must be positive, and 0 < delta < 1.")
        
        self.epsilon = epsilon
        self.delta = delta
        self.GS = GS
        self.tol = tol

    def _Phi(self, t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def _caseA(self, epsilon, s):
        return self._Phi(sqrt(epsilon * s)) - exp(epsilon) * self._Phi(-sqrt(epsilon * (s + 2.0)))

    def _caseB(self, epsilon, s):
        return self._Phi(-sqrt(epsilon * s)) - exp(epsilon) * self._Phi(-sqrt(epsilon * (s + 2.0)))

    def _doubling_trick(self, predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def _binary_search(self, predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    def add_noise(self, data):
        delta_thr = self._caseA(self.epsilon, 0.0)

        if self.delta == delta_thr:
            alpha = 1.0
        else:
            if self.delta > delta_thr:
                def predicate_stop_DT(s):
                    return self._caseA(self.epsilon, s) >= self.delta
                def function_s_to_delta(s):
                    return self._caseA(self.epsilon, s)
                def predicate_left_BS(s):
                    return function_s_to_delta(s) > self.delta
                def function_s_to_alpha(s):
                    return sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)
            else:
                def predicate_stop_DT(s):
                    return self._caseB(self.epsilon, s) <= self.delta
                def function_s_to_delta(s):
                    return self._caseB(self.epsilon, s)
                def predicate_left_BS(s):
                    return function_s_to_delta(s) < self.delta
                def function_s_to_alpha(s):
                    return sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

            def predicate_stop_BS(s):
                return abs(function_s_to_delta(s) - self.delta) <= self.tol

            s_inf, s_sup = self._doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = self._binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)

        sigma = alpha * self.GS / sqrt(2.0 * self.epsilon)
        noise = torch.normal(mean=0, std=sigma, size=data.shape).to(data.device)
        return data + noise

class StaircaseMechanism:
    def __init__(self, epsilon, sensitivity, gamma=None, random_state=None):
        """
        Initializes the Staircase mechanism with the given privacy parameters.

        :param epsilon: Privacy parameter (epsilon > 0).
        :param sensitivity: Sensitivity of the mechanism.
        :param gamma: Tuning parameter for the mechanism (0 <= gamma <= 1). If None, defaults to 1 / (1 + exp(epsilon/2)).
        :param random_state: Controls the randomness of the mechanism for reproducibility.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.gamma = self._check_gamma(gamma)
        self.random_state = random_state

    def _check_gamma(self, gamma):
        """
        Validates the gamma parameter for the Staircase mechanism.

        :param gamma: Tuning parameter for the mechanism.
        :return: Validated gamma value.
        """
        if gamma is None:
            gamma = 1 / (1 + np.exp(self.epsilon / 2))  # Default gamma value

        if not isinstance(gamma, (int, float)):
            raise TypeError("Gamma must be numeric")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma must be in [0,1]")
        
        return float(gamma)

    def _check_epsilon_delta(self, delta):
        """
        Validates the epsilon and delta parameters for the Staircase mechanism.

        :param delta: Privacy relaxation parameter delta.
        :return: True if the parameters are valid.
        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return True

    def _bias(self, value):
        """
        Bias function for the Staircase mechanism.

        :param value: The value to calculate the bias for.
        :return: Bias for the value.
        """
        return 0.0

    def _variance(self, value):
        """
        Variance function for the Staircase mechanism (not implemented).

        :param value: The value to calculate the variance for.
        :raises: NotImplementedError
        """
        raise NotImplementedError("Variance calculation is not implemented for the Staircase mechanism.")

    def add_noise(self, values):
        """
        Randomises the tensor of values using the Staircase mechanism.

        :param values: The tensor of values to randomise.
        :return: Noisy, randomised tensor.
        """
        # Set the random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Ensure that random variables are on the same device as `values`
        device = values.device

        # Generate the random variables
        sign = torch.where(torch.rand_like(values) < 0.5, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
        geometric_rv = torch.tensor(np.random.geometric(1 - np.exp(-self.epsilon), size=values.shape), device=device) - 1
        unif_rv = torch.rand_like(values, device=device)
        binary_rv = torch.where(torch.rand_like(values, device=device) < self.gamma / (self.gamma + (1 - self.gamma) * np.exp(-self.epsilon)),
                                torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        # Apply the staircase mechanism to randomise the tensor of values
        noisy_values = values + sign * ((1 - binary_rv) * ((geometric_rv + self.gamma * unif_rv) * self.sensitivity) +
                                       binary_rv * ((geometric_rv + self.gamma + (1 - self.gamma) * unif_rv) * self.sensitivity))

        return noisy_values
    
if __name__ == "__main__":
    mech = GaussianMechanism(epsilon=3, delta=1/50000, sensitivity=1.0)
    print(mech.sigma)