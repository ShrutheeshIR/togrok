import torch


class AdamCustom(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.98),
        eps = 1e-8,
        weight_decay = 0.1,
    ):
        # invalid input errors
        if lr<= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps<= 0.0:
            raise ValueError(f"eps needs to be a positive value")
        
        # define a dictionary for hyperparameters
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super(AdamCustom, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step for GD.
        """
        # Iterate through each group
        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']  
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Iterate through each individual parameter in the group.
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                # State Initialization to save parameter information over iterations
                if len(state) == 0:
                    state['step'] = 0
                    # m_t: Biased first moment estimate (like momentum)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_t: Biased second raw moment estimate (adaptive learning rate)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Get state variables for the current parameter
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Increment step counter t
                state['step'] += 1
                t = state['step']
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # --- Core Adam Logic ---

                # 1. Update the biased estimates for m_t and v_t
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # 2. Calculate the NUMERATOR: The bias-corrected first moment (m_hat_t)
                bias_correction1 = 1 - beta1 ** t
                numerator = exp_avg / bias_correction1
                
                # 3. Calculate the DENOMINATOR: The bias-corrected second moment (sqrt(v_hat_t)) + eps
                bias_correction2 = 1 - beta2 ** t
                # First, get v_hat_t
                v_hat_t = exp_avg_sq / bias_correction2
                # Then, calculate the full denominator
                denominator = v_hat_t.sqrt().add(eps)

                # 4. Calculate the final update amount
                update_step = numerator / denominator

                # 5. Apply the final update to the parameter
                # new_weight = old_weight - lr * (numerator / denominator)
                p.add_(-lr*update_step)



class SGDCustom(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGDCustom, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-lr)


class LBFGSCustom(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(LBFGSCustom, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        # LBFGS is a second-order optimization algorithm that approximates the inverse Hessian matrix to perform updates.
        # Implementing it here

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                # LBFGS logic would go here, but it's quite complex and typically relies on line search and curvature information.
                # For simplicity, we will just perform a basic gradient step here as a placeholder.
                p.add_(p.grad, alpha=-lr)

class BTLSCustom(torch.optim.Optimizer):
    def __init__(self, params, model, loss_fn, gamma=0.5, c1=0.1, weight_decay = 0.001):
        defaults = dict(model=model, loss_fn=loss_fn, gamma=gamma, c1=c1, weight_decay=weight_decay)
        super(BTLSCustom, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, x, y):
        # Implementation of back tracking linesearch in the gradient direction. reduces alpha parameter until wolfe conditions are met
        # save the old params
        alpha = 1
        while True:
            for group in self.param_groups:
                old_params = []
                model = group['model']
                loss_fn = group['loss_fn']
                c1 = group['c1']
                weight_decay = group['weight_decay']
                L0 = loss_fn(model(x), y)
                Lg = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    Lg += -torch.flatten(p.grad) @ torch.flatten(p.grad).T / (torch.linalg.norm(p.grad) ** 2)
                # compute L(alpha)
                La = L0 + c1 * alpha * Lg
                # print(f"La: {La}")

                # update params
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    p.add_(d_p, alpha = -alpha)
                    old_params.append(p)
                
                # recompute loss after param update
                loss_a = loss_fn(model(x), y)
                # print(f"loss_a: {loss_a}")

                # if wolfe is good, return
                if loss_a < La:
                    # print(alpha)
                    return
                
                # otherwise, reset params and try again with new alpha
                for i in range(len(group["params"])):
                    if group['params'][i] is None:
                        continue
                    group['params'][i] = old_params[i]
            alpha *= 0.5


# class AdamBLTSCustom(torch.optim.Optimizer):
#     def __init__(
#         self,
#         params,
#         loss_fn, 
#         lr = 1e-3,
#         betas = (0.9, 0.98),
#         eps = 1e-8,
#         weight_decay = 0.1,
#         gamma=0.5, 
#         c1=0.1,
#     ):
#         # invalid input errors
#         if lr<= 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if eps<= 0.0:
#             raise ValueError(f"eps needs to be a positive value")
        
#         # define a dictionary for hyperparameters
#         defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay, loss_fn=loss_fn, gamma=gamma, c1=c1)
#         self.max_num_attempts = 10
#         super(AdamBLTSCustom, self).__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, x):
#         """
#         Performs a single optimization step for GD.
#         """
#         # Iterate through each group
#         alpha = 1
#         num_attempts = 0
#         while num_attempts < self.max_num_attempts:
#             num_attempts += 1

#             for group in self.param_groups:

#                 lr = group['lr']
#                 beta1 = group['beta1']  
#                 beta2 = group['beta2']
#                 eps = group['eps']
#                 weight_decay = group['weight_decay']

#                 # Iterate through each individual parameter in the group.
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue

#                     state = self.state[p]
#                     # State Initialization to save parameter information over iterations
#                     if len(state) == 0:
#                         state['step'] = 0
#                         # m_t: Biased first moment estimate (like momentum)
#                         state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                         # v_t: Biased second raw moment estimate (adaptive learning rate)
#                         state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                     # To perform backtracking line search, we would need to compute the loss after applying the update and check if it satisfies the Armijo condition. If not, we would reduce alpha and try again.
#                     # for adam we simply cannot just use the norm of the grad due to the momentum term
#                     # we will compute 

#                     # Get state variables for the current parameter
#                     exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
#                     # Increment step counter t
#                     state['step'] += 1
#                     t = state['step']
#                     if weight_decay != 0:
#                         p.mul_(1 - lr * weight_decay)

#                     # --- Core Adam Logic ---

#                     # 1. Update the biased estimates for m_t and v_t
#                     exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
#                     exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

#                     # 2. Calculate the NUMERATOR: The bias-corrected first moment (m_hat_t)
#                     bias_correction1 = 1 - beta1 ** t
#                     numerator = exp_avg / bias_correction1
                    
#                     # 3. Calculate the DENOMINATOR: The bias-corrected second moment (sqrt(v_hat_t)) + eps
#                     bias_correction2 = 1 - beta2 ** t
#                     # First, get v_hat_t
#                     v_hat_t = exp_avg_sq / bias_correction2
#                     # Then, calculate the full denominator
#                     denominator = v_hat_t.sqrt().add(eps)

#                     # 4. Calculate the final update amount
#                     update_step = numerator / denominator




#                     # 5. Apply the final update to the parameter
#                     # new_weight = old_weight - lr * (numerator / denominator)
#                     p.add_(-lr*update_step)


class SecondOrderAdamCustom(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.98),
        eps = 1e-8,
        weight_decay = 0.1,
    ):
        # invalid input errors
        if lr<= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps<= 0.0:
            raise ValueError(f"eps needs to be a positive value")
        
        # define a dictionary for hyperparameters
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        super(SecondOrderAdamCustom, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step for GD.
        """
        # Iterate through each group
        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']  
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Iterate through each individual parameter in the group.
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                # State Initialization to save parameter information over iterations
                if len(state) == 0:
                    state['step'] = 0
                    # m_t: Biased first moment estimate (like momentum)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_t: Biased second raw moment estimate (adaptive learning rate)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['prev_param'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Get state variables for the current parameter
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Increment step counter t
                state['step'] += 1
                t = state['step']
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # --- Core Adam Logic ---

                # 1. Update the biased estimates for m_t and v_t
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # 2. Calculate the NUMERATOR: The bias-corrected first moment (m_hat_t)
                bias_correction1 = 1 - beta1 ** t
                numerator = exp_avg / bias_correction1
                
                # 3. Calculate the DENOMINATOR: The bias-corrected second moment (sqrt(v_hat_t)) + eps
                bias_correction2 = 1 - beta2 ** t
                # First, get v_hat_t
                v_hat_t = exp_avg_sq / bias_correction2
                # Then, calculate the full denominator
                denominator = v_hat_t.sqrt().add(eps)

                # 4. Calculate the final update amount
                update_step = numerator / denominator

                # now calculate the second order correction using the previous gradient and parameter values
                lbfgs_step = torch.zeros_like(p)
                if t > 1:
                    prev_grad = state['prev_grad']
                    prev_param = state['prev_param']

                    # compute the difference in gradients and parameters
                    yi = exp_avg - prev_grad
                    si = p - prev_param

                    w = si.reshape(-1).dot(yi.reshape(-1))
                    if w > 0:
                        # Compute the entire l-bfgs step
                        
                        # Now let's compute the entries of the different parts of the matrix
                        alpha = yi.reshape(-1).dot(si.reshape(-1))/yi.reshape(-1).dot(yi.reshape(-1))
                        M11 = 1/(w)**2*(w + alpha*yi.reshape(-1).dot(yi.reshape(-1)))
                        M12 = -1/w
                        sg = si.reshape(-1).dot(exp_avg.reshape(-1))
                        yg = alpha*yi.reshape(-1).dot(exp_avg.reshape(-1))
                        TE = M11*sg + M12*yg
                        BE = M12*(sg)
                        value = si.reshape(-1)* TE + alpha*yi.reshape(-1)*BE
                        value = alpha*(exp_avg.reshape(-1)) + value
                        lbfgs_step = value.view_as(p)
                        p.add_(-lr*lbfgs_step)
                    # else:
                    #     p.add_(-lr*update_step)


                
                # else:
                    # 5. Apply the final update to the parameter
                    # new_weight = old_weight - lr * (numerator / denominator)
                p.add_(-lr*update_step)

                # store current grad and param for next step
                state['prev_grad'] = exp_avg.clone()
                state['prev_param'] = p.clone()

