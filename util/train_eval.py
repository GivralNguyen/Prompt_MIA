import torch
import numpy as np
from util.dp_mech import GaussianMechanism


def _mixup_data(data, target, alpha):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    index = torch.randperm(data.size(0), device=data.device)
    mixed_data = lam * data + (1.0 - lam) * data[index, :]
    return mixed_data, target, target[index], lam


def _apply_gradient_sparsification(model, sparse_ratio):
    # sparse_ratio is the fraction to prune (set to zero) by magnitude.
    sparse_ratio = float(max(0.0, min(1.0, sparse_ratio)))
    if sparse_ratio <= 0.0:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if grad.numel() <= 1:
            continue
        keep_num = int(grad.numel() * (1.0 - sparse_ratio))
        if keep_num <= 0:
            grad.zero_()
            continue
        if keep_num >= grad.numel():
            continue
        flat_abs = grad.abs().reshape(-1)
        threshold = torch.topk(flat_abs, k=keep_num, largest=True).values[-1]
        grad.mul_((grad.abs() >= threshold).to(grad.dtype))


def _apply_gradient_perturbation(model, sigma):
    sigma = float(max(0.0, sigma))
    if sigma <= 0.0:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.add_(torch.randn_like(param.grad), alpha=sigma)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def scaffold_tuning_gradients(model, server_controls, client_controls):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad.data += (server_controls[name] - client_controls[name])

def cal_sensitivity(clip, dataset_size):
    return clip / dataset_size

def cal_delta(dataset_size):
    return 1/dataset_size

def train(model, data_loader, optimizer, dp_mechanism, eps, frac, epochs, loss_fun,
          device='cuda', reduce_sim_scalar=0.01, mask=None,
          defense='none', mixup_alpha=0.2, sparse_ratio=0.0,
          perturb_sigma=0.0, sampling_proportion=1.0):
    model.train()
    loss_all = 0
    total = 0
    accuracy = 0
    delta = 1/50000
    # print(len(data_loader.dataset))
    # print(delta)
    current_lr = get_lr(optimizer)
    # if dp_mechanism == 'Laplace':
    #     mech = LaplaceMechanism(epsilon=eps, sensitivity=1.0)
    if dp_mechanism == 'Gaussian':
        mech = GaussianMechanism(epsilon=eps, delta=delta, sensitivity=1.0)
    # if dp_mechanism == 'AnalyticGaussian':
    #     mech = AnalyticGaussianMechanism(epsilon=eps, delta=delta, GS=1.0, tol=1e-12)
    # if dp_mechanism == 'Staircase':
    #     mech = StaircaseMechanism(epsilon=eps, sensitivity=1.0)
    sampling_proportion = float(max(0.0, min(1.0, sampling_proportion)))
    max_steps = len(data_loader)
    if sampling_proportion < 1.0:
        max_steps = max(1, int(len(data_loader) * sampling_proportion))

    processed_steps = 0
    for step_idx, (data, target) in enumerate(data_loader):
        if step_idx >= max_steps:
            break
        # batch_size = data.size(0)
        data = data.to(device).float()
        target = target.to(device).long()

        use_mixup = defense in ['mix_up', 'mixup']
        if use_mixup:
            data, target_a, target_b, lam = _mixup_data(data, target, mixup_alpha)
        
        # Apply Local Differential Privacy (LDP) noise
        if dp_mechanism in ['Laplace', 'Gaussian', 'AnalyticGaussian', 'Staircase']:
            data = mech.add_noise(data)
            # print(torch.norm(data_-data, p=2)) # get norm of this
            # data = data_

        output = model(data)
        if isinstance(output, tuple):
            reduce_sim, logits = output
            if mask is not None:
                not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                not_mask = torch.tensor(not_mask).to(device).long()
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            if use_mixup:
                loss = lam * loss_fun(logits, target_a) + (1.0 - lam) * loss_fun(logits, target_b)
            else:
                loss = loss_fun(logits, target)
            loss = loss + reduce_sim_scalar*(reduce_sim)
            # print(loss,reduce_sim)
        else:
            if mask is not None:
                not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                not_mask = torch.tensor(not_mask).to(device).long()
                output = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
            if use_mixup:
                loss = lam * loss_fun(output, target_a) + (1.0 - lam) * loss_fun(output, target_b)
            else:
                loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        if isinstance(output, tuple):
            pred = logits.data.max(1)[1]
        else:
            pred = output.data.max(1)[1]
        if use_mixup:
            accuracy += (lam * pred.eq(target_a.view(-1)).sum().item()
                         + (1.0 - lam) * pred.eq(target_b.view(-1)).sum().item())
        else:
            accuracy += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()

        if defense == 'sparse':
            _apply_gradient_sparsification(model, sparse_ratio)
        if defense == 'perturb':
            _apply_gradient_perturbation(model, perturb_sigma)

        optimizer.step()
        processed_steps += 1

        #if isinstance(output, tuple):
            #torch.nn.utils.clip_grad_norm_(filter(lambda p : p.requires_grad, server_model.parameters()), 0.0)

    effective_steps = max(1, processed_steps)
    return current_lr, loss_all/effective_steps, accuracy/total

def evaluate(model, data_loader, loss_fun, device='cuda'):
    model.eval()
    loss_all = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            if isinstance(output, tuple):
                loss = loss_fun(output[1], target)
                loss = loss #+ 0.01*output[0]
            else:
                loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            if isinstance(output, tuple):
                pred = output[1].data.max(1)[1]
            else:
                pred = output.data.max(1)[1]
            accuracy += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), accuracy/total
    
def evaluate_mask(model, data_loader, loss_fun, device='cuda', class_mask=None):
    model.eval()
    loss_all = 0
    total = 0
    accuracy = 0
    len_data_loader = 0
    with torch.no_grad():
        for i in range(len(data_loader)):
            mask = class_mask[i]
            for data, target in data_loader[i]:
                data = data.to(device).float()
                target = target.to(device).long()
                output = model(data)
                if isinstance(output, tuple):
                    reduce_sim, logits = output
                    not_mask = np.setdiff1d(np.arange(model.num_classes), mask)
                    not_mask = torch.tensor(not_mask).to(device).long()
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    loss = loss_fun(logits, target)
                    loss = loss #+ 0.01*output[0]
                else:
                    loss = loss_fun(output, target)
                loss_all += loss.item()
                total += target.size(0)
                if isinstance(output, tuple):
                    pred = logits.data.max(1)[1]
                else:
                    pred = output.data.max(1)[1]
                accuracy += pred.eq(target.view(-1)).sum().item()
            len_data_loader += len(data_loader[i])
    return loss_all/len_data_loader, accuracy/total

