from typing import Dict, Optional, Tuple, List

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


PROBE_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "To be, or not to be, that is the question.",
    "Mixture-of-Experts routing can be student-aware.",
]


def are_tokenizers_compatible(t_tok: PreTrainedTokenizerBase, s_tok: PreTrainedTokenizerBase) -> bool:
    if type(t_tok) is not type(s_tok):
        return False
    if getattr(t_tok, "vocab_size", None) != getattr(s_tok, "vocab_size", None):
        return False
    # quick behavioral check
    for s in PROBE_STRINGS:
        if t_tok.encode(s) != s_tok.encode(s):
            return False
    return True


def build_text_datasets(
    dataset_name: str,
    dataset_config_name: Optional[str],
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    split_train: str = "train",
    split_validation: str = "validation",
    num_proc: Optional[int] = None,
) -> Tuple[Dict, Optional[Dict], bool]:
    ds = load_dataset(dataset_name, dataset_config_name)

    same_tok = are_tokenizers_compatible(teacher_tokenizer, student_tokenizer)

    if same_tok:
        tokenizer = teacher_tokenizer  # shared

        def tokenize_fn(examples):
            return tokenizer(examples["text"])  # expects a text field

        tokenized = ds.map(tokenize_fn, batched=True, num_proc=num_proc, remove_columns=["text"])

        def group_texts(examples):
            # concatenate and split into blocks
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            result = {}
            for k, t in concatenated.items():
                t = t[:total_length]
                result[k] = [t[i : i + block_size] for i in range(0, total_length, block_size)]
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized.map(group_texts, batched=True, num_proc=num_proc)

        train_dataset = lm_datasets[split_train]
        eval_dataset = lm_datasets[split_validation] if split_validation in lm_datasets else None
        return train_dataset, eval_dataset, True
    else:
        # Keep raw text. Collator will tokenize per-batch for both tokenizers.
        return ds[split_train], (ds[split_validation] if split_validation in ds else None), False


class DualTokenizerCollator:
    def __init__(self, teacher_tok: PreTrainedTokenizerBase, student_tok: PreTrainedTokenizerBase, block_size: int):
        self.tok_t = teacher_tok
        self.tok_s = student_tok
        self.block = block_size
        # Ensure padding tokens
        for tok in (self.tok_t, self.tok_s):
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    def _prep(self, tok: PreTrainedTokenizerBase, texts: List[str]):
        enc = tok(
            texts,
            max_length=self.block + 1,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'][:, : self.block]
        attn = enc['attention_mask'][:, : self.block]
        # ensure at least one real token per row
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        import torch
        for b in range(input_ids.size(0)):
            if attn[b].sum().item() == 0:
                attn[b, 0] = 1
                input_ids[b, 0] = eos_id
        labels = input_ids.clone()
        # shift labels left by one within the block
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        # mask pads in labels
        labels = labels.masked_fill(attn == 0, -100)
        # build offsets padded/truncated to block
        # offsets is a list of list of pairs
        offsets = enc['offset_mapping']
        B, L = input_ids.size()
        off_t = torch.full((B, self.block, 2), -1, dtype=torch.long)
        for b in range(B):
            row = offsets[b][: self.block]
            for i, (s, e) in enumerate(row):
                off_t[b, i, 0] = s
                off_t[b, i, 1] = e
            # ensure first offset exists if we injected eos
            if off_t[b].eq(-1).all() and attn[b, 0] == 1:
                off_t[b, 0, 0] = 0
                off_t[b, 0, 1] = 1
        return input_ids, attn, labels, off_t

    def __call__(self, batch):
        texts = [ex['text'] for ex in batch]
        ti, ta, tl, toff = self._prep(self.tok_t, texts)
        si, sa, sl, soff = self._prep(self.tok_s, texts)
        return {
            'teacher_input_ids': ti,
            'teacher_attention_mask': ta,
            'teacher_labels': tl,
            'teacher_offsets': toff,
            'student_input_ids': si,
            'student_attention_mask': sa,
            'student_labels': sl,
            'student_offsets': soff,
        }
