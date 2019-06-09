import numpy as np
from numba import njit

@njit
def newton(x_0_arr, a_arr, mu,
           tol = 1e-7, max_iter = 1000):
    """
    Newton method for equation x - x_0 + a x^mu = 0, x >= 0
    """
    res = np.empty(len(x_0_arr), dtype = np.float_)
    for i in range(len(x_0_arr)):
        x_0 = x_0_arr[i]
        a = a_arr[i]
        if x_0 <= 0:
            res[i] = 0
            continue
        x = min(x_0, (x_0 / a) ** (1 / mu))
        for it in range(max_iter):
            x_next = x - f(x, x_0, a, mu) / der_f(x, x_0, a, mu)
            if x_next <= 0:
                x_next = 0.1 * x
            x = x_next
            if np.abs(f(x, x_0, a, mu)) < tol:
                break
        res[i] = x
    return res

@njit
def f(x, x_0, a, mu):
    return x - x_0 + a * x ** mu

@njit
def der_f(x, x_0, a, mu):
    return 1.0 + a * mu * x ** (mu - 1)

class ProxH:
    def __init__(self, freeflowtimes, capacities, rho = 10.0, mu = 0.25):  
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities
        
    def __call__(self, point, A, u_start = None):
        #print('argmin called...' + 'u_start = ' + str(u_start))
        if self.mu_value == 0:
            return np.maximum(point - A * self.capacities, self.freeflowtimes)
        elif self.mu_value == 1:
            pass
        elif self.mu_value == 0.5:
            pass
        elif self.mu_value == 0.25:
            pass
        
        self.A = A
        if u_start is None:
            u_start = 2.0 * self.freeflowtimes
        x = newton(x_0_arr = (point - self.freeflowtimes) / (self.rho_value * self.freeflowtimes),
                   a_arr = A * self.capacities / (self.rho_value * self.freeflowtimes),
                   mu = self.mu_value)
        argmin = (1 + self.rho_value * x) * self.freeflowtimes
        #print('my result argmin = ' + str(argmin))
        return argmin
    
    
"""
class PhiSmallSolver:
    #""
    Function:
       phi(t) = alpha (Phi(y^0) + <grad Phi(y), t - y> + h(t)) + 1/2 ||t - y||^2_2
    #""
    
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

