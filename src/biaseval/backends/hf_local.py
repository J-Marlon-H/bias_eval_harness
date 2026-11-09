"""Hugging Face causal-language-model backend."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from biaseval.backends.base import ModelBackend
from biaseval.backends.tokenization import encode_answer_candidates


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if (
            device.type == "mps"
            and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available())
        ):
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    cpu_backend = getattr(torch.backends, "cpu", None)
    if cpu_backend is not None and getattr(cpu_backend, "has_bf16", False):
        return torch.bfloat16
    return torch.float32


class HuggingFaceBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str | None = None,
        token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.dtype = getattr(torch, dtype) if dtype else _dtype_for(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype, token=token
        ).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._scoring_method = "sequence_logprob_space_aware"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"loglik": True, "generate": True}

    @property
    def scoring_method(self) -> str:
        return self._scoring_method

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "hf_local",
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "scoring_method": self.scoring_method,
        }

    @torch.inference_mode()
    def score_options(self, prompts: list[str], n_options: int) -> list[list[float]]:
        results: list[list[float]] = []
        for prompt in prompts:
            prompt_ids, candidate_ids = encode_answer_candidates(
                self.tokenizer, prompt, n_options
            )
            sequences = [prompt_ids + ids for ids in candidate_ids]
            max_length = max(map(len, sequences))
            input_ids = torch.full(
                (len(sequences), max_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, sequence in enumerate(sequences):
                input_ids[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
                attention_mask[row, :len(sequence)] = 1
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)
            scores = []
            for row, continuation in enumerate(candidate_ids):
                start = len(prompt_ids) - 1
                positions = torch.arange(start, start + len(continuation), device=self.device)
                target_ids = input_ids[row, len(prompt_ids):len(prompt_ids) + len(continuation)]
                token_scores = log_probs[row, positions, target_ids]
                scores.append(float(token_scores.sum().cpu()))
            results.append(scores)
        return results

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        outputs = []
        for prompt in prompts:
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = generated[0, encoded["input_ids"].shape[1]:]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs
