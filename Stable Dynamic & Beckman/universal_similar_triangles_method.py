from math import sqrt
import numpy as np
from history import History

def universal_similar_triangles_method(oracle, prox, primal_dual_oracle,
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
    
    A_prev = 0.0
    y_start = u_prev = t_prev = np.copy(t_start)
    A = u = t = y = None
    
    grad_sum = None
    grad_sum_prev = np.zeros(len(t_start))

    flows_weighted = primal_dual_oracle.get_flows(y_start) 
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows_weighted, y_start)
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
            
            alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_prev / L_value)
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            grad_y = oracle.grad(y)
            flows = primal_dual_oracle.get_flows(y) #grad() is called here
            grad_sum = grad_sum_prev + alpha * grad_y
            u = prox(grad_sum / A, y_start, 1.0 / A)
            t = (alpha * u + A_prev * t_prev) / A

            left_value = (oracle.func(y) + np.dot(grad_y, t - y) + 
                          0.5 * alpha / A * eps_abs) - oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - y)**2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2
                    
        A_prev = A
        L_value /= 2
        
        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum
        flows_weighted = (flows_weighted * (A - alpha) + flows * alpha ) / A
        
        primal, dual, duality_gap, state_msg = primal_dual_oracle(flows_weighted, t)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap, inner_iters_num)
        if verbose and (it_counter % verbose_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Inner iterations number: {:d}'.format(inner_iters_num))
            print(state_msg, flush = True)
        if crit():
            success = True
            break
            
    result = {'times': t, 'flows': flows_weighted,
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

#print('Dijkstra elapsed time: {:.0f} sec'.format(oracle.auto_oracles_time))

#criteria: stable dynamic 'dual_threshold' AND 'primal_threshold', 'dual_rel' AND 'primal_rel'. 

#beckman : + 'dual_gap_rel', 'dual_gap_threshold', 'primal_threshold', 'primal_rel'

#criteria: 'star_solution_residual',

#practice: 'dual_rel'


#     if crit_name == 'dual_gap_rel':
#         def crit():
#             nonlocal duality_gap, duality_gap_init, eps
#             return duality_gap < eps * duality_gap_init
#     if crit_name == 'dual_rel':
#         def crit():
#             nonlocal dual_func_history, eps
#             l = len(dual_func_history)
#             return dual_func_history[l // 2] - dual_func_history[-1] \
#                    < eps * (dual_func_history[0] - dual_func_history[-1])
#     if crit_name == 'primal_rel':
#         def crit():
#             nonlocal primal_func_history, eps
#             l = len(primal_func_history)
#             return primal_func_history[l // 2] - primal_func_history[-1] \
#                    < eps * (primal_func_history[0] - primal_func_history[-1])