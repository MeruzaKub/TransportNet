import numpy as np
from history import History

def universal_gradient_descent_function(phi_big_oracle, prox_h, primal_dual_oracle,
                                        t_start, L_init = None, max_iter = 1000,
                                        eps = 1e-5, eps_abs = None, verbose_step = 100,
                                        verbose = False, save_history = False):    
    iter_step = verbose_step
    L_value = L_init if L_init is not None else np.linalg.norm(phi_big_oracle.grad(t_start))
    A = 0.0
    t_prev = np.copy(t_start)
    t = None
    grad_sum = np.zeros(len(t_start))

    flows_weighted = - phi_big_oracle.grad(t_start)
    t_weighted = np.copy(t_start)
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows_weighted, t_weighted)
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap', 'inner_iters')
        history.update(0, primal, dual, duality_gap_init, 0)
    if verbose:
        print(state_msg)
    if eps_abs is None:
        eps_abs = eps * duality_gap_init
    
    success = False
    inner_iters_num = 0
    
    for it_counter in range(1, max_iter+1):
        while True:
            inner_iters_num += 1
            
            alpha = 1 / L_value
            phi_grad_t = phi_big_oracle.grad(t_prev)
            t = prox_h(t_prev - alpha * phi_grad_t, alpha)

            left_value = (phi_big_oracle.func(t_prev) + np.dot(phi_grad_t, t - t_prev) + 
                          0.5 * eps_abs) - phi_big_oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - t_prev)**2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2
                    
        L_value /= 2
        
        t_prev = t
        t_weighted = (t_weighted * A + t * alpha) / (A + alpha)
        A += alpha
        grad_sum += alpha * phi_grad_t
        flows_weighted = - grad_sum / A
        
        primal, dual, duality_gap, state_msg  = primal_dual_oracle(flows_weighted, t_weighted)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap, inner_iters_num)
        if verbose and (it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Inner iterations number: {:d}'.format(inner_iters_num))
            print(state_msg, flush = True)
        if duality_gap < eps_abs:
            success = True
            break
            
    result = {'times': t_weighted, 'flows': flows_weighted,
              'iter_num': it_counter,
              'res_msg' : 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('Result: ' + result['res_msg'], 'Total iters: ' + str(it_counter))
        print(state_msg)
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
    return result
