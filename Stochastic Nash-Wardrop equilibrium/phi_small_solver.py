import numpy as np
import scipy
from scipy.special import expit
from scipy.optimize import newton
from newton_optimization import newton_raphson_method
from numba import jitclass

class PhiSmallSolver:
    """
    Function:
       phi(t) = alpha (Phi(y^0) + <grad Phi(y), t - y> + h(t)) + 1/2 ||t - y||^2_2
    """
    
    def __init__(self, freeflowtimes, capacities, rho = 10.0, mu = 0.25):  
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities
        
        self.alphas_sum = 0.0
        self.alpha_phi_big_grad_sum = np.zeros(self.links_number)
    

    def update(self, alpha_new, phi_big_grad):
        #print('phi_small called. update... ' + 'y_parameter_new = ' + str(y_parameter_new))
        self.alphas_sum += alpha_new
        self.alpha_phi_big_grad_sum += alpha_new * phi_big_grad


    def undo_update(self, alpha_new, phi_big_grad):
        #print('phi_small called. update... ' + 'y_parameter_new = ' + str(y_parameter_new))
        self.alphas_sum -= alpha_new
        self.alpha_phi_big_grad_sum -= alpha_new * phi_big_grad


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
                                               args = (self.alpha_phi_big_grad_sum[link_index],
                                                       self.capacities[link_index],
                                                       self.freeflowtimes[link_index]),
                                               tolerance = 1e-7 * self.freeflowtimes[link_index],
                                               max_iter = 1000)
            argmin[link_index] = y_min
        #print('my result argmin = ' + str(argmin))
        return argmin
    

    def grad_component(self, t_value, alpha_phi_big_grad_sum, capacity, freeflowtime):
        #print('t_value = ' + str(t_value))
        return alpha_phi_big_grad_sum + \
               self.alphas_sum * capacity * \
               np.power((t_value - freeflowtime) / (self.rho_value * freeflowtime), self.mu_value)\
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

