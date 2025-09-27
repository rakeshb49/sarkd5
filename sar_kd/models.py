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
    use_fp16: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)
    student_tok = AutoTokenizer.from_pretrained(student_model_name, use_fast=True)
    # Ensure padding is defined for batching
    for tok in (teacher_tok, student_tok):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    # Determine actual model dtype - use FP16 for models if specified to save memory
    model_dtype = torch.float16 if use_fp16 else torch_dtype
    print(f"Loading models with dtype: {model_dtype}")

    # trust_remote_code True to support custom MoE impls
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=model_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=model_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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

    # Move models to device with memory-efficient loading
    print(f"Moving teacher model to {device}...")
    teacher.to(device)
    print(f"Moving student model to {device}...")
    student.to(device)

    # Enable gradient checkpointing for both models to save memory
    if hasattr(teacher, "gradient_checkpointing_enable"):
        try:
            teacher.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for teacher model")
        except Exception as e:
            print(f"Could not enable gradient checkpointing for teacher: {e}")

    if gradient_checkpointing_student and hasattr(student, "gradient_checkpointing_enable"):
        try:
            student.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for student model")
        except Exception as e:
            print(f"Could not enable gradient checkpointing for student: {e}")

    # Print memory usage after loading models
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    return teacher, student, teacher_tok, student_tok
