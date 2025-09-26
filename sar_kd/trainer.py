import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedModel

from .losses import (
    ce_loss,
    kd_kl_loss,
    router_anchor_l2,
    router_entropy_bonus,
    router_load_balance_loss,
)
from .router_utils import GatingStats, collect_router_linears, freeze_all_but_router, snapshot_router_state


@dataclass
class SARConfig:
    temperature: float = 2.0
    alpha_kd: float = 0.9
    alpha_ce: float = 0.1
    router_anchor_weight: float = 1e-4
    router_load_balance_weight: float = 1e-3
    router_entropy_weight: float = 1e-4
    student_lr: float = 1e-4
    router_lr: float = 5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


class SARDistiller:
    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        device: torch.device,
        config: SARConfig,
        tokenizers_compatible: bool = True,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.config = config
        self.same_tok = tokenizers_compatible

        self.router_linears = collect_router_linears(self.teacher)
        if len(self.router_linears) == 0:
            print("[WARN] No router-like linear layers found. Proceeding without router training.")
        self.router_params = freeze_all_but_router(self.teacher, self.router_linears)
        self.router_init = snapshot_router_state(self.router_linears)

        self.gating_stats = GatingStats(device)
        self.gating_stats.register_on(self.router_linears)

        # Optional teacher->student hidden projector if dims mismatch or tokenizers differ
        self.h_proj = None
        if not self.same_tok or (getattr(self.teacher.config, 'hidden_size', None) != getattr(self.student.config, 'hidden_size', None)):
            in_dim = getattr(self.teacher.config, 'hidden_size', None)
            out_dim = getattr(self.student.config, 'hidden_size', None)
            if in_dim is not None and out_dim is not None:
                self.h_proj = nn.Linear(in_dim, out_dim, bias=True).to(self.device)

        # Create optimizers with separate param groups
        param_groups = [
            {"params": self.student.parameters(), "lr": config.student_lr, "weight_decay": config.weight_decay},
            {"params": self.router_params, "lr": config.router_lr, "weight_decay": 0.0},
        ]
        if self.h_proj is not None:
            param_groups.append({"params": self.h_proj.parameters(), "lr": config.student_lr, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(param_groups)

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        steps: int,
        grad_accum_steps: int = 1,
        eval_every: int = 0,
        save_callback=None,
        log_callback=None,
    ):
        self.teacher.train()  # only router params are trainable
        self.student.train()

        step = 0
        total_tokens = 0
        running_loss = 0.0
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        while step < steps:
            for batch in train_loader:
                if step >= steps:
                    break
                # Handle batch structure depending on tokenizer compatibility
                if 'teacher_input_ids' in batch:
                    input_ids = None
                    labels = None
                    attention_mask = None
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    if 'teacher_input_ids' in batch:
                        # dual path
                        t_ids = batch['teacher_input_ids'].to(self.device)
                        t_mask = batch['teacher_attention_mask'].to(self.device)
                        s_ids = batch['student_input_ids'].to(self.device)
                        s_mask = batch['student_attention_mask'].to(self.device)
                        t_out = self.teacher(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                        s_out = self.student(input_ids=s_ids, attention_mask=s_mask, output_hidden_states=True)

                        # Hidden-state MSE alignment using offsets
                        t_h = t_out.hidden_states[-1]  # [B, Lt, Ht]
                        s_h = s_out.hidden_states[-1]  # [B, Ls, Hs]
                        if self.h_proj is not None:
                            t_h_proj = self.h_proj(t_h)
                        else:
                            t_h_proj = t_h

                        # build alignment: average teacher hidden over tokens overlapping each student token span
                        t_off = batch['teacher_offsets'].to(self.device)
                        s_off = batch['student_offsets'].to(self.device)

                        mse_terms = []
                        count = 0
                        B = s_h.size(0)
                        for b in range(B):
                            ti = t_h_proj[b]
                            si = s_h[b]
                            to = t_off[b]
                            so = s_off[b]
                            # determine valid student positions
                            valid_s = (so[:, 0] >= 0) & (so[:, 1] >= 0)
                            for i in torch.nonzero(valid_s, as_tuple=False).flatten().tolist():
                                s_start, s_end = so[i].tolist()
                                if s_end <= s_start:
                                    continue
                                # find teacher positions that overlap
                                t_valid = (to[:, 0] >= 0) & (to[:, 1] >= 0)
                                t_sel = []
                                for j in torch.nonzero(t_valid, as_tuple=False).flatten().tolist():
                                    t_start, t_end = to[j].tolist()
                                    if t_end <= t_start:
                                        continue
                                    # overlap if intervals intersect
                                    if not (t_end <= s_start or t_start >= s_end):
                                        t_sel.append(j)
                                if not t_sel:
                                    continue
                                t_mean = ti[t_sel].mean(dim=0)
                                mse_terms.append((si[i] - t_mean).pow(2).mean())
                                count += 1
                        kd = torch.stack(mse_terms).mean() if mse_terms else torch.tensor(0.0, device=self.device)
                        if torch.isnan(kd):
                            kd = torch.tensor(0.0, device=self.device)

                        # CE on student's own labels
                        s_logits = s_out.logits[:, :-1, :]
                        s_labels = batch['student_labels'][:, 1:].to(self.device)
                        if (s_labels != -100).any():
                            ce = ce_loss(s_logits, s_labels)
                        else:
                            ce = torch.tensor(0.0, device=self.device)
                        if torch.isnan(ce):
                            ce = torch.tensor(0.0, device=self.device)
                    else:
                        # same-tokenizer path
                        t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask)

                        # shift logits and labels by one for next-token prediction
                        teacher_logits = t_out.logits[:, :-1, :]
                        student_logits = s_out.logits[:, :-1, :]
                        target_labels = labels[:, 1:]

                        kd = kd_kl_loss(student_logits, teacher_logits, self.config.temperature)
                        ce = ce_loss(student_logits, target_labels)

                    # Router regularization
                    current_router_state = {}
                    for name, lin in self.router_linears:
                        for pname, p in lin.named_parameters(recurse=False):
                            current_router_state[f"{name}.{pname}"] = p
                    anchor = router_anchor_l2(current_router_state, self.router_init) * self.config.router_anchor_weight

                    gate_probs = self.gating_stats.pop_and_reset()
                    lb = router_load_balance_loss(gate_probs) * self.config.router_load_balance_weight
                    ent = router_entropy_bonus(gate_probs) * self.config.router_entropy_weight

                    loss = self.config.alpha_kd * kd + self.config.alpha_ce * ce + anchor + lb + ent
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = torch.tensor(0.0, device=self.device)

                loss = loss / grad_accum_steps
                if loss.requires_grad:
                    scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 and loss.requires_grad:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.router_params, self.config.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * grad_accum_steps
                if 'teacher_input_ids' in batch:
                    total_tokens += int((s_mask.sum()).item())
                else:
                    total_tokens += int((attention_mask.sum() if attention_mask is not None else input_ids.numel()).item())

                step += 1
                if log_callback and step % max(1, grad_accum_steps) == 0:
                    log_callback({
                        "step": step,
                        "train_loss": running_loss / max(1, step),
                        "tokens": total_tokens,
                    })

                if eval_every and eval_loader is not None and step % eval_every == 0:
                    ppl = self.evaluate(eval_loader)
                    if log_callback:
                        log_callback({"step": step, "val_ppl": ppl})
                    if save_callback:
                        save_callback(step, self.student, self.router_linears)

        if save_callback:
            save_callback(step, self.student, self.router_linears)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        self.student.eval()
        total_nll = 0.0
        total_tokens = 0
        for batch in eval_loader:
            if 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(self.device)
                labels = batch['student_labels'].to(self.device)
                attention_mask = batch.get('student_attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            out = self.student(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :]
            target = labels[:, 1:]
            vocab = logits.size(-1)
            nll = nn.functional.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), ignore_index=-100, reduction='sum')
            total_nll += nll.item()
            valid = (target != -100).sum().item()
            total_tokens += valid
        self.student.train()
        ppl = math.exp(total_nll / max(1, total_tokens))
        return ppl