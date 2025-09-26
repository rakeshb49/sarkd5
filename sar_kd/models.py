from typing import Tuple

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_teacher_student(
    teacher_model_name: str,
    student_model_name: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing_student: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)
    student_tok = AutoTokenizer.from_pretrained(student_model_name, use_fast=True)
    # Ensure padding is defined for batching
    for tok in (teacher_tok, student_tok):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    # trust_remote_code True to support custom MoE impls
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )

    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )

    # Ensure pad_token_id in configs and disable cache when training
    try:
        teacher.config.pad_token_id = teacher_tok.pad_token_id
        teacher.config.use_cache = False
    except Exception:
        pass
    try:
        student.config.pad_token_id = student_tok.pad_token_id
        student.config.use_cache = False
    except Exception:
        pass

    teacher.to(device)
    student.to(device)

    if gradient_checkpointing_student and hasattr(student, "gradient_checkpointing_enable"):
        try:
            student.gradient_checkpointing_enable()
        except Exception:
            pass

    return teacher, student, teacher_tok, student_tok