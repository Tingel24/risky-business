import json
import os
import re
import sys
from typing import Dict, Any, List
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from tqdm import tqdm

from faith_shop import shuffle_choices, FIXED_CASE_WORDS


def swap_str(m: str, a: str, b: str) -> str:
    w1_esc = re.escape(a)
    w2_esc = re.escape(b)

    patterns = sorted([w1_esc, w2_esc], key=len, reverse=True)
    pattern_str = "|".join(patterns)

    pattern = re.compile(rf"(?<!\w)({pattern_str})(?!\w)", re.IGNORECASE)

    def replacement_logic(match):
        matched_text = match.group(0)

        if matched_text.lower() == a.lower():
            target = b
        else:
            target = a

        for fixed_word in FIXED_CASE_WORDS:
            if target.lower() == fixed_word.lower():
                return fixed_word

        if matched_text.isupper():
            return target.upper()
        elif matched_text[0].isupper():
            return target.title()
        elif matched_text.islower():
            return target.lower()
        else:
            return target

    return pattern.sub(replacement_logic, m)


def apply_new_order(data: Dict[str, Any], target_order: List[str]) -> None:
    if "order" not in data or "messages" not in data:
        raise KeyError("Data dictionary must contain order and messages keys")

    if not isinstance(target_order, list) or len(target_order) != 3:
        raise ValueError("Target order must be a list of exactly 3 elements")

    current_order: List[str] = list(data["order"])
    messages: str = data["messages"]
    identifiers: List[str] = ["id_x", "id_y", "id_z"]

    for target_idx in range(len(target_order)):
        target_item: str = target_order[target_idx]

        if current_order[target_idx] != target_item:
            current_idx: int = current_order.index(target_item)

            messages = swap_str(messages, identifiers[target_idx], identifiers[current_idx])

            current_order[current_idx] = current_order[target_idx]
            current_order[target_idx] = target_item

    data["messages"] = messages
    data["order"] = list(target_order)


def format_for_dpo(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    prompt, target_order = shuffle_choices(example["entry"])

    apply_new_order(example["chosen"], target_order)
    apply_new_order(example["rejected"], target_order)

    system_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
    formatted_system = tokenizer.apply_chat_template(
        system_msg,
        tokenize=False
    )

    clean_system_content = formatted_system.replace("<|im_start|>system\n", "").replace("<|im_end|>\n", "").strip()

    prompt_msgs = [
        {"role": "system", "content": clean_system_content},
        {"role": "user", "content": prompt}
    ]

    return {
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": example["chosen"]["messages"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]["messages"]}]
    }


def run_dpo_training(
        dataset_path: str = "../data/filtered_dpo_train.jsonl",
        model_id: str = "Qwen/QwQ-32B",
        output_dir: str = "results/qwq_aligned_model",
        dry_run: bool = False,
) -> None:
    setup_steps: int = 4

    with tqdm(total=setup_steps, desc="Training Setup Progress", position=0, leave=True) as pbar:

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formatted_records = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                clean_record = format_for_dpo(record, tokenizer)
                formatted_records.append(clean_record)

        dataset = Dataset.from_list(formatted_records)
        pbar.update(1)

        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        pbar.update(1)

        if dry_run:
            print(f"\nDRY RUN Dataset formatted successfully with {len(train_dataset)} training and {len(eval_dataset)} eval examples.")
            print(f"DRY RUN Columns passed to trainer {train_dataset.column_names}")
            print("DRY RUN Sample prompt output:")
            print(train_dataset[0]["prompt"])
            return

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={'': local_rank},
            local_files_only=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        pbar.update(1)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            bf16=True,
            logging_first_step=True,
            remove_unused_columns=False,
            optim="paged_adamw_32bit",
            beta=0.1,
            max_length=4096,
            max_prompt_length=2048
        )

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            args=training_args,
            peft_config=peft_config,
        )
        pbar.update(1)

    print("\nStarting alignment training loop")
    dpo_trainer.train()

    print(f"Saving final aligned model weights to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO alignment training.")
    parser.add_argument("--dry-run", action="store_true", help="Test the dataset pipeline without starting the GPU training loop")
    args = parser.parse_args()

    DATASET_FILE = "../data/filtered_dpo_train.jsonl"
    BASE_MODEL = "/mnt/ceph-ssd/workspaces/ws/ag_gipp/u25472-models/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
    OUTPUT_PATH = "../results/qwq_aligned_model"

    with open("../data/shop_dataset_system_prompt.md", 'r', encoding='utf-8') as file:
        SYSTEM_PROMPT = file.read()

    run_dpo_training(
        dataset_path=DATASET_FILE,
        model_id=BASE_MODEL,
        output_dir=OUTPUT_PATH,
        dry_run=args.dry_run
    )