import numpy as np
import scipy
from scipy.special import expit
from scipy.optimize import newton
from newton_optimization import newton_raphson_method
    
class PhiSmallSolver:
    """
    Function:
       phi(t) = alpha (Phi(y^0) + <grad Phi(y), t - y> + h(t)) + 1/2 ||t - y||^2_2
    """
#rho_value = 10.0 ?
    def __init__(self, phi_big_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25):
        self.phi_big_oracle = phi_big_oracle
        
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities
        
        self.alphas_sum = 0.0
        self.alpha_phi_big_grad_sum = np.zeros(self.links_number)
        self.y_parameter_current = self.freeflowtimes
        
    def update(self, alpha_new, y_parameter_new = None):
        #print('phi_small called. update... ' + 'y_parameter_new = ' + str(y_parameter_new))
        self.alphas_sum += alpha_new
        if y_parameter_new is not None:
            self.y_parameter_current = y_parameter_new
        self.alpha_phi_big_grad_sum += alpha_new * \
                                       self.phi_big_oracle.grad(self.y_parameter_current)

    def argmin_function(self, u_start = None):
        #print('argmin called...' + 'u_start = ' + str(u_start))
        if u_start is None:
            u_start = 2.0 * self.freeflowtimes
            #print('u_start = ' + str(u_start))
        argmin = np.zeros(self.links_number)
        for link_index in range(0, self.links_number):
            argmin[link_index], msg = newton_raphson_method(x_start = u_start[link_index],
                                                            boundary_value = self.freeflowtimes[link_index],
                                                            grad_func = self.grad_component,
                                                            hess_func = self.hess_diagonal_component,
                                                            args = (self.alpha_phi_big_grad_sum[link_index],
                                                                    self.capacities[link_index],
                                                                    self.freeflowtimes[link_index]),
                                                            tolerance = 1e-6, max_iter=1000)
        #print('my result argmin = ' + str(argmin))
        return argmin
    

    def grad_component(self, t_value, alpha_phi_big_grad_sum, capacity, freeflowtime):
        #print('t_value = ' + str(t_value))
        return alpha_phi_big_grad_sum + \
               self.alphas_sum * capacity *\
               np.power((t_value - freeflowtime) / (self.rho_value * freeflowtime), self.mu_value) \
               + t_value - freeflowtime

    #hess_diagonal element returned. Hess is diagonal matrix
    def hess_diagonal_component(self, t_value, alpha_phi_big_grad_sum, capacity, freeflowtime):
        return self.alphas_sum * self.mu_value * capacity * \
               np.power((t_value - freeflowtime) / (self.rho_value * freeflowtime), self.mu_value - 1.0) / \
               (self.rho_value * freeflowtime) + 1.0
        

"""      
    def argmin_function(self, u_start = None):
        if u_start is None:
            u_start = np.zeros(self.links_number)
        argmin = np.zeros(self.links_number)
        for link_index in range(0, self.links_number):
            
            argmin[link_index] = newton(self.grad_component, x0 = u_start[link_index],
                                        fprime = self.hess_diagonal_component,
                                        args = (self.alpha_phi_big_grad_sum[link_index],
                                                self.capacities[link_index],
                                                self.freeflowtimes[link_index],
                                                self.freeflowtimes[link_index]),
                                        tol=1e-6, maxiter=500, fprime2=None)
            
        out_of_set_elements_indices = np.where(argmin < self.freeflowtimes)
        argmin[out_of_set_elements_indices] = self.freeflowtimes[out_of_set_elements_indices]
        return argmin
        
"""    

