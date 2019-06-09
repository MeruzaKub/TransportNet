import numpy as np
from scipy.optimize import newton
from newton_optimization import newton_raphson_method

class UnivGradPhiSmallSolver:
    """
    Function:
       y = t^k, 
       phi(t) = Phi(t^k) + <grad Phi(t^k), t - t^k> + h(t)) + L^{k+1}  1/2 ||t - t^k||^2_2
       
       V(t, t^k) = 1/2 ||t - t^k||^2_2
       
       equivalent
       min <grad Phi(t^k), t - t^k> + h(t)) + L^{k+1}  1/2 ||t - t^k||^2_2
    """
    
    def __init__(self, freeflowtimes, capacities, rho = 10.0, mu = 0.25):  
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities
        
        self.L_new = 0.0
        self.phi_big_grad = None
        self.t_current = None
    

    def update(self, L_new, phi_big_grad, t_current):
        #print('phi_small called. update... ' + 'y_parameter_new = ' + str(y_parameter_new))
        self.L_new = L_new
        self.phi_big_grad = phi_big_grad
        self.t_current = t_current


    def argmin_function(self, u_start = None):
        #print('argmin called...' + 'u_start = ' + str(u_start))
        if self.mu_value == 0:
            pass
        elif self.mu_value == 1:
            pass
        elif self.mu_value == 0.5:
            pass
        elif self.mu_value == 0.25:
            pass
        
        if u_start is None:
            u_start = 2.0 * self.freeflowtimes
            #print('u_start = ' + str(u_start))
        argmin = np.empty(self.links_number)
        for link_index in range(self.links_number):
            y_min, msg = newton_raphson_method(x_start = u_start[link_index],
                                               boundary_value = self.freeflowtimes[link_index],
                                               grad_func = self.grad_component,
                                               hess_func = self.hess_diagonal_component,
                                               args = (self.phi_big_grad[link_index],
                                                       self.capacities[link_index],
                                                       self.freeflowtimes[link_index],
                                                       self.t_current[link_index]),
                                               tolerance = 1e-7 * self.freeflowtimes[link_index],
                                               max_iter = 1000)
            argmin[link_index] = y_min
        #print('my result argmin = ' + str(argmin))
        return argmin
    

    def grad_component(self, t_value, phi_big_grad, capacity, freeflowtime, t_current):
        #print('t_value = ' + str(t_value))
        return phi_big_grad + \
               capacity * \
               np.power((t_value - freeflowtime) / (self.rho_value * freeflowtime), self.mu_value)\
               + self.L_new * (t_value - t_current)

    #hess_diagonal element returned. Hess is diagonal matrix
    def hess_diagonal_component(self, t_value, phi_big_grad, capacity, freeflowtime, t_current):
        return self.mu_value * capacity * \
               np.power((t_value - freeflowtime) / (self.rho_value * freeflowtime), self.mu_value - 1.0) / \
               (self.rho_value * freeflowtime) + self.L_new
        
        
