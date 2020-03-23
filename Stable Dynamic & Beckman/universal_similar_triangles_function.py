from math import sqrt
import numpy as np
from history import History


#criteria: stable dynamic 'dual_threshold' AND 'primal_threshold', 'dual_rel' AND 'primal_rel'. 

#beckman : + 'dual_gap_rel', 'dual_gap_threshold', 'primal_threshold', 'primal_rel'

#criteria: 'star_solution_residual',

#practice: 'dual_rel'


def universal_similar_triangles_function(phi_big_oracle, prox_h, primal_dual_oracle,
                                         t_start, L_init = None, max_iter = 1000,
                                         eps = 1e-5, eps_abs = None, 
                                         verbose = False, save_history = False):    
    
    iter_step = 100
    if L_init is not None:
        L_value = L_init
    else:
        L_value = np.linalg.norm(phi_big_oracle.grad(t_start))
    
    A_prev = 0.0
    y_start = u_prev = t_prev = np.copy(t_start)
    A = u = t = y = None
    
    grad_sum = None
    grad_sum_prev = np.zeros(len(t_start))
    flows_weighted = np.zeros(len(t_start))

    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap', 'inner_iters')
    
    success = False
    inner_iters_num = 0
    
    for it_counter in range(1, max_iter+1):
        while True:
            alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_prev / L_value)
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            phi_grad_y = phi_big_oracle.grad(y)
            grad_sum = grad_sum_prev + alpha * phi_grad_y
            u = prox_h(y_start - grad_sum, A, u_start = u_prev)
            t = (alpha * u + A_prev * t_prev) / A

            inner_iters_num += 1
            if inner_iters_num == 1:
                flows_weighted = - grad_sum / A
                duality_gap_init = primal_dual_oracle.duality_gap(t, flows_weighted)
                if eps_abs is None:
                    eps_abs = eps * duality_gap_init
                
                if verbose:
                    print('Primal_init = {:g}'.format(primal_dual_oracle.primal_func_value(flows_weighted)))
                    print('Dual_init = {:g}'.format(primal_dual_oracle.dual_func_value(t)))
                    print('Duality_gap_init = {:g}'.format(duality_gap_init))

            left_value = (phi_big_oracle.func(y) + np.dot(phi_grad_y, t - y) + 
                          0.5 * alpha / A * eps_abs) - phi_big_oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - y)**2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2
                inner_iters_num += 1

                    
        A_prev = A
        L_value /= 2
        
        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum
        flows_weighted = - grad_sum / A
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows_weighted)
        dual_func_value = primal_dual_oracle.dual_func_value(t)
        duality_gap = primal_dual_oracle.duality_gap(t, flows_weighted)
        
        if save_history:
            history.update(it_counter, primal_func_value, dual_func_value, duality_gap, inner_iters_num)
        
        if duality_gap < eps_abs:
            success = True
            break
        
        if verbose and (it_counter == 1 or it_counter % iter_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Inner iterations number: {:d}'.format(inner_iters_num))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dual_func_value = {:g}'.format(dual_func_value))
            print('Duality_gap = {:g}'.format(duality_gap))
            print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init), flush=True)
            
            
    result = {'times': t,
              'flows': flows_weighted,
              'iter_num': it_counter}
    
    if save_history:
        result['history'] = history.dict
    
    if success:
        result['res_msg'] = 'success'
    else:
        result['res_msg'] = 'iterations number exceeded'
        
    if verbose:
        if success:
            print('\nSuccess! Iterations number: ' + str(it_counter))
        else:
            print('\nIterations number exceeded!')
        print('Primal_func_value = {:g}'.format(primal_func_value))
        print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init))
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
        
    return result


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