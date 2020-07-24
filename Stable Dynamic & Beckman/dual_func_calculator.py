import numpy as np

class PrimalDualCalculator:
    def __init__(self, phi_big_oracle, h_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25, base_flows = None):
        self.links_number = len(freeflowtimes)
        self.rho = rho
        self.mu = mu
        self.freeflowtimes = freeflowtimes #\bar{t}
        self.capacities = capacities       #\bar{f}
        
        self.phi_big_oracle = phi_big_oracle
        self.h_oracle = h_oracle
        self.dual_gap_init = None
        if mu == 0:
            if base_flows is None:
                raise TypeError("Admissible flows should be given")
            elif np.any(base_flows < 0) or np.any(base_flows >= capacities):
                raise ValueError("Admissible flows should be non-negative and less than capacities")
            else:
                self.base_flows = base_flows
                self.alpha = 1 - np.max(base_flows / capacities)
        
    def __call__(self, flows, times):
        gap = self.duality_gap(times, flows)
        primal = self.primal_func_value(flows)
        dual = self.dual_func_value(times)
        if self.dual_gap_init is None:
            self.dual_gap_init = gap
            state_msg = 'Primal_init = {:g}'.format(primal) + \
                         '\nDual_init = {:g}'.format(dual) + \
                         '\nDuality_gap_init = {:g}'.format(self.dual_gap_init)
        else:
            state_msg = 'Primal_func_value = {:g}'.format(primal) + \
                         '\nDual_func_value = {:g}'.format(dual) + \
                         '\nDuality_gap = {:g}'.format(gap) + \
                         '\nDuality_gap / Duality_gap_init = {:g}'.format(gap / self.dual_gap_init)
        return primal, dual, gap, state_msg
    
    
    def dual_func_value(self, times):
        return self.phi_big_oracle.func(times) + self.h_oracle.func(times)
    
    def primal_func_value(self, flows):
        return self.h_oracle.conjugate_func(flows)
    
    def duality_gap(self, times, flows):
        if self.mu > 0:
            return self.dual_func_value(times) + self.primal_func_value(flows)
        else:
            beta = max(0, np.max(flows / self.capacities) - 1)
            admissible_flows = (beta * self.base_flows + self.alpha * flows) / (self.alpha + beta)
            return self.dual_func_value(times) + self.primal_func_value(admissible_flows)
    
    def get_flows(self, times):
        return - self.phi_big_oracle.grad(times)
    
    #for Frank-Wolfe algorithm
    def get_times(self, flows):
        return self.freeflowtimes * (1.0 + self.rho * np.power(flows / self.capacities, 1.0 / self.mu))
