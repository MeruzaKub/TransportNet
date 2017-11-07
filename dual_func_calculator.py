import numpy as np

class DualFuncCalculator:
    def __init__(self, phi_big_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25):
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities
        
        self.phi_big_oracle = phi_big_oracle
        self.t_current = None
        self.dual_func_value = None
    
    def compute_value(self, t_parameter):
        #print('dual func...')
        #print('t_current_dual = ' + str(t_parameter))
        assert(not np.any(np.isnan(t_parameter)))
        if self.t_current is None or np.any(self.t_current != t_parameter):
            h_function_value = np.sum(1.0 / (1.0 + self.mu_value) * self.capacities * \
                                      np.power(np.maximum(t_parameter - self.freeflowtimes, 0.0) / \
                                               (self.rho_value * self.freeflowtimes), self.mu_value) * \
                                      (t_parameter - self.freeflowtimes))
            self.dual_func_value = self.phi_big_oracle.func(t_parameter) + h_function_value
            self.t_current = t_parameter
        return self.dual_func_value
