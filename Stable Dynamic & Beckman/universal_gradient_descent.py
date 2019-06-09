from math import sqrt
import numpy as np

def universal_gradient_descent_function(phi_big_oracle, prox_h, primal_dual_oracle,
                                         t_start, L_init = None, max_iter = 1000,
                                         eps = 1e-5, eps_abs = None, verbose = False):
    iter_step = 100
    if L_init is not None:
        L_value = L_init
    else:
        L_value = np.linalg.norm(phi_big_oracle.grad(t_start))
    
    A = 0.0
    t_prev = np.copy(t_start)
    t = None
    
    grad_sum = np.zeros(len(t_start))
    flows_weighted = np.zeros(len(t_start))
    
    duality_gap_init = None

    primal_func_history = []
    dual_func_history = []
    inner_iters_history = []
    duality_gap_history = []
        
    success = False
    
    for it_counter in range(1,max_iter+1):
        inner_iters_num = 1
        while True:
            alpha = 1 / L_value

            phi_grad_t = phi_big_oracle.grad(t_prev)
            t = prox_h(t_prev - alpha * phi_grad_t, alpha)

            if it_counter == 1 and inner_iters_num == 1:
                flows_weighted = - phi_grad_t
                duality_gap_init = primal_dual_oracle.duality_gap(t, flows_weighted)
                if eps_abs is None:
                    eps_abs = eps * duality_gap_init
                
                if verbose:
                    print('Primal_init = {:g}'.format(primal_dual_oracle.primal_func_value(flows_weighted)))
                    print('Dual_init = {:g}'.format(primal_dual_oracle.dual_func_value(t)))
                    print('Duality_gap_init = {:g}'.format(duality_gap_init))

            left_value = (phi_big_oracle.func(t_prev) + 
                          np.dot(phi_grad_t, t - t_prev) + 
                          0.5 * eps_abs) - phi_big_oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - t_prev)**2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2
                inner_iters_num += 1

        L_value /= 2
        
        t_prev = t
        A += alpha
        grad_sum += alpha * phi_grad_t
        flows_weighted = - grad_sum / A
        
        primal_func_value = primal_dual_oracle.primal_func_value(flows_weighted)
        dual_func_value = primal_dual_oracle.dual_func_value(t)
        duality_gap = primal_dual_oracle.duality_gap(t, flows_weighted)
        
        primal_func_history.append(primal_func_value)
        dual_func_history.append(dual_func_value)
        duality_gap_history.append(duality_gap)
        inner_iters_history.append(inner_iters_num)
        
        #if duality_gap < eps_abs:
        #    success = True
        #    break
        
        if verbose and (it_counter == 1 or (it_counter) % iter_step == 0):
            print('\nIterations number: ' + str(it_counter))
            print('Inner iterations number: ' + str(inner_iters_num))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dual_func_value = {:g}'.format(dual_func_value))
            print('Duality_gap = {:g}'.format(duality_gap))
            print('Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init))
            
            
    result = {'times': t,
              'flows': flows_weighted,
              'iter_num': it_counter,
              'duality_gap_history': duality_gap_history,
              'inner_iters_history': inner_iters_history,
              'primal_func_history': primal_func_history,
              'dual_func_history': dual_func_history,
             }
    
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