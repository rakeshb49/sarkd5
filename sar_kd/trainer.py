import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.data.data_collator import DataCollatorForLanguageModeling

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
    use_scheduler: bool = False
    warmup_steps: int = 100
    scheduler_type: str = 'cosine'
    total_steps: int = 1000


class SARDistiller:
    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        device: torch.device,
        config: SARConfig,
        tokenizers_compatible: bool = True,
        router_patterns: Optional[List[str]] = None,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.config = config
        self.same_tok = tokenizers_compatible

        self.router_linears = collect_router_linears(self.teacher, custom_patterns=router_patterns, verbose=True)
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

        # Initialize learning rate scheduler if requested
        self.scheduler = None
        if config.use_scheduler:
            from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
            if config.scheduler_type == 'linear':
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=config.warmup_steps,
                    num_training_steps=config.total_steps
                )
            elif config.scheduler_type == 'cosine':
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=config.warmup_steps,
                    num_training_steps=config.total_steps
                )
            print(f"Using {config.scheduler_type} learning rate scheduler with {config.warmup_steps} warmup steps")

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
        self.teacher.eval()  # disable dropout but router params still trainable via requires_grad=True
        self.student.train()

        step = 0
        total_tokens = 0
        running_loss = 0.0
        # Enable scaler for mixed precision training, especially important with FP16 parameters
        use_scaler = torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Print scaler status for user information
        if torch.cuda.is_available():
            has_fp16_params = any(p.dtype == torch.float16 for p in list(self.student.parameters()) + list(self.router_params))
            if has_fp16_params:
                print("FP16 models detected - gradient scaler enabled to prevent gradient underflow")
            else:
                print("FP32 models detected - gradient scaler enabled for mixed precision training")
        else:
            print("CUDA not available - gradient scaler disabled")

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

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
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

                        # Vectorized hidden-state alignment
                        mse_loss = torch.tensor(0.0, device=self.device)
                        total_alignments = 0

                        B = s_h.size(0)
                        for b in range(B):
                            ti = t_h_proj[b]  # [Lt, H]
                            si = s_h[b]      # [Ls, H]
                            to = t_off[b]    # [Lt, 2]
                            so = s_off[b]    # [Ls, 2]

                            # Get valid positions
                            valid_s = (so[:, 0] >= 0) & (so[:, 1] >= 0)  # [Ls]
                            valid_t = (to[:, 0] >= 0) & (to[:, 1] >= 0)  # [Lt]

                            if not valid_s.any() or not valid_t.any():
                                continue

                            # Filter valid offsets and hidden states
                            s_valid_idx = torch.nonzero(valid_s, as_tuple=False).flatten()
                            t_valid_idx = torch.nonzero(valid_t, as_tuple=False).flatten()

                            so_valid = so[s_valid_idx]  # [n_valid_s, 2]
                            to_valid = to[t_valid_idx]  # [n_valid_t, 2]
                            si_valid = si[s_valid_idx]  # [n_valid_s, H]
                            ti_valid = ti[t_valid_idx]  # [n_valid_t, H]

                            # Vectorized overlap computation
                            # so_valid[:, 0:1] -> [n_valid_s, 1], to_valid[:, 1:2].T -> [1, n_valid_t]
                            # Broadcast to [n_valid_s, n_valid_t]
                            s_starts = so_valid[:, 0:1]  # [n_valid_s, 1]
                            s_ends = so_valid[:, 1:2]    # [n_valid_s, 1]
                            t_starts = to_valid[:, 0:1].T  # [1, n_valid_t]
                            t_ends = to_valid[:, 1:2].T    # [1, n_valid_t]

                            # Check for overlaps: intervals [s_start, s_end) and [t_start, t_end) overlap
                            # if not (t_end <= s_start or t_start >= s_end)
                            # which is equivalent to (t_end > s_start) and (t_start < s_end)
                            overlaps = (t_ends > s_starts) & (t_starts < s_ends)  # [n_valid_s, n_valid_t]

                            # For each student position, find overlapping teacher positions
                            for s_idx in range(overlaps.size(0)):
                                overlap_mask = overlaps[s_idx]  # [n_valid_t]
                                if not overlap_mask.any():
                                    continue

                                # Average teacher hidden states that overlap with this student position
                                t_overlapping = ti_valid[overlap_mask]  # [n_overlap, H]
                                t_mean = t_overlapping.mean(dim=0)      # [H]

                                # MSE loss
                                mse_loss += (si_valid[s_idx] - t_mean).pow(2).mean()
                                total_alignments += 1

                        kd = mse_loss / max(1, total_alignments) if total_alignments > 0 else torch.tensor(0.0, device=self.device)
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
                    lb = router_load_balance_loss(gate_probs, self.device) * self.config.router_load_balance_weight
                    ent = router_entropy_bonus(gate_probs, self.device) * self.config.router_entropy_weight

                    loss = self.config.alpha_kd * kd + self.config.alpha_ce * ce + anchor + lb + ent
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = torch.tensor(0.0, device=self.device)

                loss = loss / grad_accum_steps
                did_backward = False
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                    did_backward = True

                if (step + 1) % grad_accum_steps == 0:
                    if did_backward:
                        if scaler.is_enabled():
                            # Standard AMP workflow with gradient scaling
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.router_params, self.config.max_grad_norm)
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            # Manual step without scaler (CPU only)
                            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.router_params, self.config.max_grad_norm)
                            self.optimizer.step()
                    # Always zero grads at accumulation boundary
                    self.optimizer.zero_grad(set_to_none=True)

                    # Step learning rate scheduler if enabled
                    if self.scheduler is not None:
                        self.scheduler.step()

                running_loss += loss.item() * grad_accum_steps
                # Count tokens that actually contribute to the loss for accurate throughput
                if 'teacher_input_ids' in batch:
                    # For dual tokenizer mode, count valid student labels
                    s_labels = batch.get('student_labels')
                    if s_labels is not None:
                        total_tokens += int((s_labels != -100).sum().item())
                    else:
                        total_tokens += int((s_mask.sum()).item())
                else:
                    # For same tokenizer mode, count valid labels
                    if labels is not None:
                        total_tokens += int((labels != -100).sum().item())
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
