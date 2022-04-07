import numpy as np
from history import History

def universal_gradient_descent_method(oracle, prox, primal_dual_oracle,
                                      t_start, L_init = None, max_iter = 1000,
                                      eps = 1e-5, eps_abs = None, stop_crit = 'dual_gap_rel',
                                      verbose_step = 100, verbose = False, save_history = False):
    if stop_crit == 'dual_gap_rel':
        def crit():
            return duality_gap <= eps * duality_gap_init
    elif stop_crit == 'dual_gap':
        def crit():
            return duality_gap <= eps_abs
    elif stop_crit == 'max_iter':
        def crit():
            return it_counter == max_iter
    elif callable(stop_crit):
        crit = stop_crit
    else:
        raise ValueError("stop_crit should be callable or one of the following names: \
                         'dual_gap', 'dual_gap_rel', 'max iter'")
    
    L_value = L_init if L_init is not None else np.linalg.norm(oracle.grad(t_start))
    A = 0.0
    t_prev = np.copy(t_start)
    t = None

    flows_weighted = primal_dual_oracle.get_flows(t_start) 
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
            grad_t = oracle.grad(t_prev)
            flows = primal_dual_oracle.get_flows(t_prev) #grad() is called here
            t = prox(grad_t, t_prev, 1.0 / alpha)

            left_value = (oracle.func(t_prev) + np.dot(grad_t, t - t_prev) + 
                          0.5 * eps_abs) - oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - t_prev)**2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2
                    
        L_value /= 2
        
        t_prev = t
        A += alpha
        t_weighted = (t_weighted * (A - alpha) + t * alpha) / A
        flows_weighted = (flows_weighted * (A - alpha) + flows * alpha ) / A
        
        primal, dual, duality_gap, state_msg = primal_dual_oracle(flows_weighted, t_weighted)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap, inner_iters_num)
        if verbose and (it_counter % verbose_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Inner iterations number: {:d}'.format(inner_iters_num))
            print(state_msg, flush = True)
        if crit():
            success = True
            break
            
    result = {'times': t_weighted, 'flows': flows_weighted,
              'iter_num': it_counter,
              'res_msg': 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result
