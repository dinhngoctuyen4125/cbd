#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute symmetric KL divergence scores between a finetuned model (A1) and
the original pretrained model (A0) on deepseek test datasets.

Used for:
  - Stage 3 (Threshold): compute scores on D_test_U_dep / D_test_U_nondep
  - Stage 4 (Scoring): compute scores on remaining samples
"""

import os
import json
import inspect
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
from peft import LoraConfig, PeftModel
from routing_score_reducers import (
    DEFAULT_ROUTING_REDUCER_ALPHA,
    DEFAULT_ROUTING_REDUCER_BETA,
    DEFAULT_ROUTING_REDUCER_GAMMA,
    DEFAULT_ROUTING_REDUCER_TOP_M,
    ROUTING_SCORE_SEMANTICS_VERSION,
    reduce_routing_scores,
    routing_entropy_from_logp,
    routing_reducer_params,
    routing_surprisal_from_actual_tokens,
)


def load_peft_model_compat(base_model, adapter_path):
    try:
        return PeftModel.from_pretrained(base_model, adapter_path)
    except TypeError as exc:
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        allowed = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
        dropped = sorted(set(raw_cfg.keys()) - set(filtered_cfg.keys()))
        if dropped:
            print(f"[peft-load] drop unsupported adapter_config keys: {dropped} ({exc})")
        return PeftModel.from_pretrained(base_model, adapter_path, config=LoraConfig(**filtered_cfg))


def load_json_dataset(dataset_dir, split_name):
    """Load a JSON dataset from dataset_dir/split_name.json.
    Returns a DatasetDict with a 'train' split.
    """
    json_file = os.path.join(dataset_dir, f"{split_name}.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Dataset file not found: {json_file}")

    print(f"Loading dataset: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)

        if first_non_ws == "[":
            data = json.load(f)
        else:
            data = [json.loads(line.strip()) for line in f if line.strip()]

    dataset = DatasetDict({"train": Dataset.from_list(data)})
    print(f"Dataset size: {len(dataset['train'])}, columns: {dataset['train'].column_names}")
    return dataset


def format_prompt(question, raw=False):
    """Format input as prompt. raw=True for code completion (no prefix/suffix)."""
    if raw:
        return question
    return f"question: {question.strip()} answer:"


def _left_pad_tokenize(tokenizer, texts, device):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        max_len = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            max_len = 2048
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    return {k: v.to(device) for k, v in enc.items()}


def _first_eos_lengths(gen_tokens: torch.LongTensor, eos_token_id: int) -> torch.LongTensor:
    bsz, steps = gen_tokens.shape
    if steps == 0:
        return torch.zeros((bsz,), device=gen_tokens.device, dtype=torch.long)
    pos = torch.arange(steps, device=gen_tokens.device).unsqueeze(0).expand(bsz, steps)
    eos_mask = gen_tokens.eq(eos_token_id)
    eos_pos = torch.where(eos_mask, pos, torch.full_like(pos, steps))
    first = eos_pos.min(dim=1).values
    lengths = torch.where(first < steps, first + 1, torch.full_like(first, steps))
    return lengths.to(torch.long)


def _masked_reduce(token_scores: torch.Tensor, lengths: torch.LongTensor):
    bsz, steps = token_scores.shape
    if steps == 0:
        z = torch.zeros((bsz,), device=token_scores.device, dtype=token_scores.dtype)
        return z, z.clone(), z.clone()

    pos = torch.arange(steps, device=token_scores.device).unsqueeze(0).expand(bsz, steps)
    mask = pos < lengths.unsqueeze(1)
    masked = token_scores.clone()
    masked[~mask] = 0.0
    safe_len = lengths.clamp(min=1).to(token_scores.dtype)
    mean = masked.sum(dim=1) / safe_len

    big = torch.finfo(token_scores.dtype).max
    max_scores = token_scores.masked_fill(~mask, -big)
    maxv = max_scores.max(dim=1).values
    min_scores = token_scores.masked_fill(~mask, big)
    minv = min_scores.min(dim=1).values
    maxv = torch.where(lengths > 0, maxv, torch.zeros_like(maxv))
    minv = torch.where(lengths > 0, minv, torch.zeros_like(minv))
    return mean, maxv, minv


def _span_mean_max_score(token_scores: torch.Tensor, window: int) -> torch.Tensor:
    if token_scores.dim() == 1:
        token_scores = token_scores.unsqueeze(0)
    bsz, steps = token_scores.shape
    if steps == 0 or window <= 0:
        return torch.zeros((bsz,), device=token_scores.device, dtype=token_scores.dtype)
    w = min(window, steps)
    cumsum = torch.cumsum(token_scores, dim=1)
    prefix = torch.zeros((bsz, 1), device=cumsum.device, dtype=cumsum.dtype)
    cumsum = torch.cat([prefix, cumsum], dim=1)
    span_sums = cumsum[:, w:] - cumsum[:, :steps - w + 1]
    span_means = span_sums / w
    return span_means.max(dim=1).values


def compute_fixed_path_kl_batch(
    original_model,
    finetuned_model,
    prompt_input_ids: torch.LongTensor,
    prompt_attention_mask: torch.LongTensor,
    tokenizer,
    max_new_tokens: int = 32,
    symmetric: bool = False,
    escort_alpha: float = DEFAULT_ROUTING_REDUCER_ALPHA,
    escort_beta: float = DEFAULT_ROUTING_REDUCER_BETA,
    sces_gamma: float = DEFAULT_ROUTING_REDUCER_GAMMA,
    sces_top_m: int = DEFAULT_ROUTING_REDUCER_TOP_M,
    span_window: int = 4,
):
    """
    Batched fixed-path KL:
      - Use original_model greedy path (generate) to obtain sequences + per-step logits
      - Compute finetuned logits on the same sequences in one forward pass
      - Return per-sample scores (mean over generated steps, include EOS step)
    """
    with torch.no_grad():
        gen_out = original_model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    sequences = gen_out.sequences
    scores_list = list(getattr(gen_out, "scores", []) or [])
    steps = len(scores_list)

    if steps == 0:
        bsz = int(prompt_input_ids.size(0))
        empty = [
            {
                "cross_entropy": 0.0,
                "max_token_ce": 0.0,
                "min_token_ce": 0.0,
                "avg_token_ce": 0.0,
                "feis_score": 0.0,
                "cbd_weighted_kl": 0.0,
                "escort_score": 0.0,
                "sces_score": 0.0,
                "span_score": 0.0,
            }
            for _ in range(bsz)
        ]
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return texts, sequences, empty

    orig_step_logits = torch.stack(scores_list, dim=1)
    prompt_len = int(prompt_input_ids.size(1))
    start = max(prompt_len - 1, 0)

    gen_mask = torch.ones(
        (prompt_attention_mask.size(0), sequences.size(1) - prompt_len),
        device=prompt_attention_mask.device,
        dtype=prompt_attention_mask.dtype,
    )
    full_attention_mask = torch.cat([prompt_attention_mask, gen_mask], dim=1)

    with torch.no_grad():
        ft_out = finetuned_model(input_ids=sequences, attention_mask=full_attention_mask)
        ft_logits_full = ft_out.logits

    ft_step_logits = ft_logits_full[:, start : start + steps, :]

    orig_logp = F.log_softmax(orig_step_logits, dim=-1)
    ft_logp = F.log_softmax(ft_step_logits, dim=-1)
    orig_p = orig_logp.exp()

    kl_of = (orig_p * (orig_logp - ft_logp)).sum(dim=-1)
    if symmetric:
        ft_p = ft_logp.exp()
        kl_fo = (ft_p * (ft_logp - orig_logp)).sum(dim=-1)
        token_scores = 0.5 * (kl_of + kl_fo)
    else:
        token_scores = kl_of

    orig_entropy = routing_entropy_from_logp(orig_logp, probs=orig_p)
    gen_tokens = sequences[:, prompt_len:]
    orig_surprisal = routing_surprisal_from_actual_tokens(orig_logp, gen_tokens)

    lengths = _first_eos_lengths(gen_tokens, tokenizer.eos_token_id)
    mean, maxv, minv = _masked_reduce(token_scores, lengths)

    details = []
    for i in range(int(prompt_input_ids.size(0))):
        cur_len = int(lengths[i].item())
        if cur_len > 0:
            s_i = orig_surprisal[i, :cur_len]
            h_i = orig_entropy[i, :cur_len]
            kl_i = token_scores[i, :cur_len]

            feis_val = reduce_routing_scores(kl_i, "fsis", entropy=h_i, surprisal=s_i).item()
            cbd_val = reduce_routing_scores(kl_i, "cbd").item()
            escort_val = reduce_routing_scores(
                kl_i, "escort", entropy=h_i, surprisal=s_i,
                alpha=escort_alpha, beta=escort_beta,
            ).item()
            sces_val = reduce_routing_scores(
                kl_i, "sces", entropy=h_i, surprisal=s_i,
                gamma=sces_gamma, top_m=sces_top_m,
            ).item()
            span_val = _span_mean_max_score(kl_i, span_window).item()
        else:
            feis_val = cbd_val = escort_val = sces_val = span_val = 0.0

        details.append(
            {
                "cross_entropy": float(mean[i].item()),
                "max_token_ce": float(maxv[i].item()),
                "min_token_ce": float(minv[i].item()),
                "avg_token_ce": float(mean[i].item()),
                "feis_score": float(feis_val),
                "cbd_weighted_kl": float(cbd_val),
                "escort_score": float(escort_val),
                "sces_score": float(sces_val),
                "span_score": float(span_val),
            }
        )

    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return texts, sequences, details


def main():
    parser = argparse.ArgumentParser(description="Compute sym-KL scores between A0 and A1")
    parser.add_argument("--model_path", type=str, required=True, help="Finetuned model (A1) path")
    parser.add_argument("--pretrained_model_name", type=str,
                      default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Original pretrained model (A0)")
    parser.add_argument("--dataset_name", type=str, default="../Data-Collection/deepseek",
                      help="Dataset directory path")
    parser.add_argument("--dataset_split", type=str, required=True,
                      help="Dataset split name (e.g. D_test_U_dep, D_test_U_nondep)")
    parser.add_argument("--output_file", type=str, default="sym_kl_results.json", help="Output filename")
    parser.add_argument("--output_dir", type=str, default="artifacts/ce_deepseek", help="Output directory")
    parser.add_argument("--verbose", type=str, default="False", help="Print per-sample details")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Max new tokens to generate")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples to evaluate")
    parser.add_argument("--gpu_id", type=str, default="", help="GPU ID override")
    parser.add_argument("--question_key", type=str, default="probing input", help="Input field name in dataset")
    parser.add_argument("--answer_key", type=str, default="", help="Answer field name (optional, for logging)")
    parser.add_argument("--raw_prompt", action="store_true",
                      help="Use input as-is without question:/answer: prefix (for code completion)")
    parser.add_argument("--skip_samples", type=int, default=0,
                      help="Skip first N samples (for separating threshold vs scoring sets)")
    parser.add_argument("--score_reducer_alpha", type=float, default=DEFAULT_ROUTING_REDUCER_ALPHA)
    parser.add_argument("--score_reducer_beta", type=float, default=DEFAULT_ROUTING_REDUCER_BETA)
    parser.add_argument("--sces_gamma", type=float, default=DEFAULT_ROUTING_REDUCER_GAMMA)
    parser.add_argument("--sces_top_m", type=int, default=DEFAULT_ROUTING_REDUCER_TOP_M)
    parser.add_argument("--span_window", type=int, default=4)

    args = parser.parse_args()
    args.verbose = args.verbose.lower() == 'true'

    print("=" * 60)
    print("Scoring config:")
    print(f"  metric: fixed_sym_kl")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  raw_prompt: {args.raw_prompt}")
    print(f"  skip_samples: {args.skip_samples}")
    print("=" * 60)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse dataset splits (comma-separated)
    splits_to_test = [s.strip() for s in args.dataset_split.split(",") if s.strip()]
    print(f"Splits to test: {splits_to_test}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load finetuned model (A1)
    print(f"Loading finetuned model: {args.model_path}")
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_lora_model = "lora" in args.model_path.lower() or os.path.exists(adapter_config_path)

    if is_lora_model:
        print("Loading LoRA model")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_name, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=True
        )
        finetuned_model = load_peft_model_compat(base_model, args.model_path)
        finetuned_model = finetuned_model.merge_and_unload()
    else:
        print("Loading full model")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=True
        )
    finetuned_model.eval()

    # Load original model (A0)
    print(f"Loading original model: {args.pretrained_model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name, torch_dtype=torch.bfloat16,
        device_map="auto", local_files_only=True
    )
    original_model.eval()

    # Test each split
    for split_name in splits_to_test:
        print(f"\n{'='*60}")
        print(f"Testing split: {split_name}")
        print(f"{'='*60}")

        # Load dataset
        dataset = load_json_dataset(args.dataset_name, split_name)

        # Apply skip_samples
        if args.skip_samples > 0 and len(dataset["train"]) > args.skip_samples:
            dataset["train"] = dataset["train"].select(range(args.skip_samples, len(dataset["train"])))
            print(f"Skipped first {args.skip_samples} samples, remaining: {len(dataset['train'])}")

        # Apply max_samples
        max_samples = min(int(args.max_samples), len(dataset["train"]))
        if max_samples < len(dataset["train"]):
            dataset["train"] = dataset["train"].select(range(max_samples))
            print(f"Limited to {max_samples} samples")

        output_path = Path(args.output_file)
        detailed_results_file = os.path.join(args.output_dir, f"{output_path.stem}_{split_name}{output_path.suffix}")

        results = []
        examples = list(dataset["train"])
        total = len(examples)
        batch_size = max(int(args.batch_size), 1)
        print(f"Total: {total}, Batch size: {batch_size}")

        for start_idx in tqdm(range(0, total, batch_size)):
            batch = examples[start_idx : start_idx + batch_size]
            questions = [ex[args.question_key] for ex in batch]
            answers = [ex.get(args.answer_key, "") for ex in batch] if args.answer_key else [""] * len(batch)
            prompts = [format_prompt(q, raw=args.raw_prompt) for q in questions]

            enc = _left_pad_tokenize(tokenizer, prompts, device=device)
            prompt_ids = enc["input_ids"]
            prompt_mask = enc.get("attention_mask", torch.ones_like(prompt_ids))
            original_texts, _, score_details_list = compute_fixed_path_kl_batch(
                original_model=original_model,
                finetuned_model=finetuned_model,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                tokenizer=tokenizer,
                max_new_tokens=int(args.max_new_tokens),
                symmetric=True,
                escort_alpha=float(args.score_reducer_alpha),
                escort_beta=float(args.score_reducer_beta),
                sces_gamma=float(args.sces_gamma),
                sces_top_m=int(args.sces_top_m),
                span_window=int(args.span_window),
            )

            for j in range(len(batch)):
                i = start_idx + j
                score_details = score_details_list[j]
                cross_entropy = score_details["cross_entropy"]

                if args.verbose:
                    print(f"\nSample {i+1}: {questions[j][:80]}...")
                    if cross_entropy is not None:
                        print(f"  score: {cross_entropy:.4f}")

                results.append({
                    "id": i,
                    "question": questions[j],
                    "dataset_answer": answers[j],
                    "cross_entropy": cross_entropy,
                    "max_token_ce": score_details.get("max_token_ce"),
                    "feis_score": score_details.get("feis_score"),
                    "cbd_weighted_kl": score_details.get("cbd_weighted_kl"),
                    "escort_score": score_details.get("escort_score"),
                    "sces_score": score_details.get("sces_score"),
                    "span_score": score_details.get("span_score"),
                })

                # Save intermediate results every 10 samples
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    with open(detailed_results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"Saved {i+1}/{total} to {detailed_results_file}")

        # Summary statistics
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cross_entropies = [item["cross_entropy"] for item in results if item["cross_entropy"] is not None]
        avg_ce = sum(cross_entropies) / len(cross_entropies) if cross_entropies else None

        eval_results = {
            "total_samples": len(results),
            "average_sym_kl": avg_ce,
            "max_sym_kl": max(cross_entropies) if cross_entropies else None,
            "min_sym_kl": min(cross_entropies) if cross_entropies else None,
            "std_sym_kl": float(np.std(cross_entropies)) if cross_entropies else None,
            "finetuned_model_path": args.model_path,
            "original_model_path": args.pretrained_model_name,
            "dataset": f"{args.dataset_name}/{split_name}",
            "skip_samples": args.skip_samples,
            "timestamp": timestamp,
        }

        eval_output_file = os.path.join(args.output_dir, f"eval_summary_{split_name}_{timestamp}.json")
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {eval_output_file}")
        if avg_ce is not None:
            print(f"Average sym-KL: {avg_ce:.4f}")
            print(f"Max: {max(cross_entropies):.4f}, Min: {min(cross_entropies):.4f}")
            print(f"Std: {np.std(cross_entropies):.4f}")


if __name__ == "__main__":
    main()
