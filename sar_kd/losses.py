from typing import Dict

import torch
import torch.nn.functional as F


def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    T = temperature
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    loss = F.kl_div(s, t, reduction="batchmean") * (T * T)
    return loss


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    vocab = student_logits.size(-1)
    return F.cross_entropy(student_logits.view(-1, vocab), labels.view(-1), ignore_index=ignore_index)


def router_anchor_l2(current: Dict[str, torch.Tensor], initial: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=list(current.values())[0].device if current else "cpu")
    for k, cur in current.items():
        init = initial[k].to(device=cur.device, dtype=cur.dtype)
        loss = loss + F.mse_loss(cur, init)
    return loss


def router_load_balance_loss(gate_probs_by_layer: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Switch Transformers style: encourage mean usage per expert to be uniform
    if not gate_probs_by_layer:
        return torch.tensor(0.0)
    loss = None
    for name, probs in gate_probs_by_layer.items():
        # probs: [tokens, n_experts]
        if probs.numel() == 0:
            continue
        mean_usage = probs.mean(dim=0)
        n_exp = mean_usage.numel()
        l = (n_exp * (mean_usage.pow(2).sum()))
        loss = l if loss is None else (loss + l)
    if loss is None:
        loss = torch.tensor(0.0, device=list(gate_probs_by_layer.values())[0].device)
    return loss


def router_entropy_bonus(gate_probs_by_layer: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not gate_probs_by_layer:
        return torch.tensor(0.0)
    loss = None
    for _, probs in gate_probs_by_layer.items():
        # encourage higher entropy => subtract entropy => loss = -H
        p = probs.clamp(min=1e-8)
        H = -(p * p.log()).sum(dim=-1).mean()
        l = -H
        loss = l if loss is None else (loss + l)
    if loss is None:
        loss = torch.tensor(0.0, device=list(gate_probs_by_layer.values())[0].device)
    return loss