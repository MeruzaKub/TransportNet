import numpy as np
#from numba import jit

class PrimalDualCalculator:
    def __init__(self, phi_big_oracle, h_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25):
        self.links_number = len(freeflowtimes)
        self.rho = rho
        self.mu = mu
        self.freeflowtimes = freeflowtimes #\bar{t}
        self.capacities = capacities       #\bar{f}
        
        self.phi_big_oracle = phi_big_oracle
        self.h_oracle = h_oracle
        self.dual_gap_init = None
        self.A = self.entropy_weighted = 0.
    
    
    def __call__(self, flows, times, alpha = 0.):
        primal = self.primal_func_value(flows, times, alpha)
        dual = self.dual_func_value(times)
        gap = primal + dual
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
#    @jit
    def dual_func_value(self, times):
        assert(not np.any(np.isnan(times)))
        return self.phi_big_oracle.func(times) + self.h_oracle.func(times)
    
    
#    @jit
    def primal_func_value(self, flows, times, alpha):
    #upper estimate of the primal function when gamma > 0
        self.A += alpha
        if self.A == 0:
            self.entropy_weighted = self.phi_big_oracle.entropy(times)
        else:
            self.entropy_weighted = ((self.A - alpha) * self.entropy_weighted + 
                                      alpha * self.phi_big_oracle.entropy(times)) / self.A 
        return self.h_oracle.conjugate_func(flows) - self.entropy_weighted
    
    def get_flows(self, times):
        return - self.phi_big_oracle.grad(times)
