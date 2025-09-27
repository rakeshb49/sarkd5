import copy
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


DEFAULT_ROUTER_PATTERNS = [
    r"gate",              # e.g., MixtralSparseMoeBlock.gate (Linear)
    r"router",            # generic router naming
    r"moe.*gate",         # moe blocks
    r"switch.*router",
    r"expert.*router",
]

ROUTER_NAME_PATTERNS = [
    re.compile(p, flags=re.IGNORECASE)
    for p in DEFAULT_ROUTER_PATTERNS
]


def is_router_like(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lname = name.lower()
    return any(p.search(lname) for p in ROUTER_NAME_PATTERNS)


def collect_router_linears(model: nn.Module, custom_patterns: List[str] = None, verbose: bool = True) -> List[Tuple[str, nn.Linear]]:
    """
    Collect router linear layers from the model.

    Args:
        model: The model to search for router layers
        custom_patterns: Optional list of regex patterns to use instead of defaults
        verbose: Whether to log the discovered router layers

    Returns:
        List of (name, module) tuples for router layers
    """
    global ROUTER_NAME_PATTERNS

    # Use custom patterns if provided
    if custom_patterns is not None:
        patterns = [re.compile(p, flags=re.IGNORECASE) for p in custom_patterns]
        if verbose:
            print(f"Using custom router patterns: {custom_patterns}")
    else:
        patterns = ROUTER_NAME_PATTERNS
        if verbose:
            print(f"Using default router patterns: {DEFAULT_ROUTER_PATTERNS}")

    # Temporarily override global patterns for is_router_like
    original_patterns = ROUTER_NAME_PATTERNS
    ROUTER_NAME_PATTERNS = patterns

    try:
        linears: List[Tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if is_router_like(name, module):
                linears.append((name, module))

        if verbose:
            if linears:
                print(f"Found {len(linears)} router layer(s):")
                for name, module in linears:
                    print(f"  - {name}: {type(module).__name__} (in_features={module.in_features}, out_features={module.out_features})")
            else:
                print("WARNING: No router layers found! Router training will not occur.")
                print("Consider using custom patterns with --router_patterns if your model uses different naming conventions.")

        return linears
    finally:
        # Restore original patterns
        ROUTER_NAME_PATTERNS = original_patterns


def freeze_all_but_router(model: nn.Module, router_linears: List[Tuple[str, nn.Linear]]):
    for p in model.parameters():
        p.requires_grad = False
    router_params = []
    for _, lin in router_linears:
        for p in lin.parameters():
            p.requires_grad = True
            router_params.append(p)
    return router_params


def snapshot_router_state(router_linears: List[Tuple[str, nn.Linear]]) -> Dict[str, torch.Tensor]:
    snap = {}
    for name, lin in router_linears:
        for pname, p in lin.named_parameters(recurse=False):
            snap[f"{name}.{pname}"] = p.detach().clone().to(dtype=torch.float32)
    return snap


class GatingStats:
    def __init__(self, device: torch.device):
        self.device = device
        self.buffers: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_on(self, router_linears: List[Tuple[str, nn.Linear]]):
        def hook_factory(layer_name: str):
            def hook(_module, _inp, out):
                # out: pre-topk logits for experts, shape (..., n_experts)
                try:
                    logits = out
                    while logits.dim() > 2:
                        logits = logits.view(-1, logits.size(-1))
                    probs = torch.softmax(logits, dim=-1)
                    self.buffers.setdefault(layer_name, []).append(probs.detach())
                except Exception:
                    pass
            return hook

        for name, lin in router_linears:
            h = lin.register_forward_hook(hook_factory(name))
            self.hooks.append(h)

    def pop_and_reset(self) -> Dict[str, torch.Tensor]:
        agg: Dict[str, torch.Tensor] = {}
        for k, lst in self.buffers.items():
            if len(lst) == 0:
                continue
            agg[k] = torch.cat(lst, dim=0)
        self.buffers = {}
        return agg

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []
