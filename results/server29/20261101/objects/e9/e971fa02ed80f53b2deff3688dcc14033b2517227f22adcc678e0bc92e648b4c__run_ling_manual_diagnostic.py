#!/usr/bin/env python3
"""Create canonical HF Ling teacher-forced traces for SGLang closure."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data01/user2/models/Ling-3.0-tiny")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--teacher-transitions", type=int, default=128)
    args = parser.parse_args()

    os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    import torch
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    rows = rows[: args.prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    config = json.loads((Path(args.model) / "config.json").read_text())
    n_layers, group = int(config["num_hidden_layers"]), int(config["layer_group_size"])
    cutoff = n_layers // group * group
    kda_layers = [i for i in range(n_layers) if not ((i + 1) % group == 0 or i >= cutoff)]
    selected = [kda_layers[0], kda_layers[len(kda_layers) // 2], kda_layers[-1]]

    manifest = {
        "runtime": "canonical_manual_hf",
        "model": args.model,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "bfloat16",
        "selected_kda_layers": selected,
        "teacher_transitions": args.teacher_transitions,
        "prompts": [],
    }
    device = next(model.parameters()).device
    for prompt_index, row in enumerate(rows):
        prompt_dir = outdir / f"prompt_{prompt_index:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "user", "content": row["problem"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        plen = input_ids.shape[-1]
        teacher = []
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=torch.arange(plen, device=device),
                use_cache=True,
            )
            past = result.past_key_values
            for step in range(args.teacher_transitions + 1):
                states = {}
                for layer in selected:
                    state = getattr(past.layers[layer], "keys", None)
                    if state is None:
                        raise RuntimeError(f"missing KDA state at layer {layer}")
                    states[layer] = state[0].detach().cpu()
                torch.save(
                    {"logits": result.logits[:, -1].float().cpu(), "states": states},
                    prompt_dir / f"manual_step{step:04d}.pt",
                )
                next_token = int(result.logits[:, -1].argmax(-1).item())
                teacher.append(next_token)
                if step == args.teacher_transitions:
                    break
                token = torch.tensor([[next_token]], device=device)
                attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=-1)
                result = model(
                    input_ids=token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    cache_position=torch.tensor([plen + step], device=device),
                    use_cache=True,
                )
                past = result.past_key_values
        prompt_record = {
            "prompt_index": prompt_index,
            "problem_id": row["problem_id"],
            "messages": messages,
            "prompt_text": prompt_text,
            "input_ids": input_ids[0].cpu().tolist(),
            "input_ids_hash": hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest(),
            "teacher_token_ids": teacher,
            "teacher_forced_transitions": args.teacher_transitions,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        save_json(prompt_dir / "prompt.json", prompt_record)
        manifest["prompts"].append({k: prompt_record[k] for k in (
            "prompt_index", "problem_id", "input_ids_hash", "teacher_forced_transitions"
        )})
        print(f"manual diagnostic {prompt_index + 1}/{len(rows)} complete", flush=True)
    save_json(outdir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
