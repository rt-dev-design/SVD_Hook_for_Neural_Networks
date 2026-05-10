import torch

def effective_step_stats(optimizer):
    effective_step_size = {}
    group_index = 0

    for group in optimizer.param_groups:
        steps = []
        lr = group['lr']

        for p in group['params']:
            if p.grad is None:
                continue

            state = optimizer.state[p]

            # --- Adam / AdamW ---
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                m = state['exp_avg']
                v = state['exp_avg_sq']
                eps = group.get('eps', 1e-8)
                step = (lr * m.abs() / (v.sqrt() + eps)).flatten()

            # --- SGD (with or without momentum) ---
            elif 'momentum_buffer' in state:
                buf = state['momentum_buffer']
                step = (lr * buf.abs()).flatten()
            else:
                # plain SGD (no momentum)
                step = (lr * p.grad.abs()).flatten()

            steps.append(step.detach().cpu())

        # if len(steps) == 0:
        #     continue

        all_steps = torch.cat(steps)
        effective_step_size[group_index] = {
            'mean': all_steps.mean().item(),
            'max': all_steps.max().item(),
            'median': all_steps.median().item()
        }
        group_index += 1

    return effective_step_size

def calculate_grad_norm(model):
    with torch.no_grad():
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.detach().norm().item()**2
        total_grad_norm = total_grad_norm**0.5
        return total_grad_norm