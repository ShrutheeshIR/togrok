import torch


class AdamCustom(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        beta1 = 0.9,
        beta2 = 0.98,
        eps = 1e-8,
        weight_decay = 0.1,
    ):
        # invalid input errors
        if lr<= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps<= 0.0:
            raise ValueError(f"eps needs to be a positive value")
        
        # define a dictionary for hyperparameters
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2 , eps=eps, weight_decay=weight_decay)
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