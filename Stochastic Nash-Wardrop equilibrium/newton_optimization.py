from numba import jit

#Newton rhapson 1D method for the case of x >= x_boundary
@jit
def newton_raphson_method(x_start, boundary_value,
                          grad_func, hess_func, args,
                          tolerance = 1e-6, max_iter = 100):

    x_current = max(x_start, boundary_value + tolerance)
        
    grad_start_value = grad_func(x_current, *args)
    
    for counter in range(max_iter):
        grad_value = grad_func(x_current, *args)      
        #stop criteria
        #if abs(grad_value) <= tolerance * abs(grad_start_value):
        #    return x_current, 'success'
        
        #calculating descent direction
        hess_value = hess_func(x_current, *args)
        x_next = x_current - grad_value / hess_value
        
        if x_next <= boundary_value:
            x_next = 0.5 * (x_current + boundary_value)
        
        if abs(x_next - x_current) < tolerance:
            return x_next, 'success'
        
        x_current = x_next
    return x_current, 'iterations_exceeded' 
