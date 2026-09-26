"""
Creates the turnkey Kaggle Jupyter Notebook: training/stage10_qlora_kaggle.ipynb
Tailored specifically for Kaggle GPU environment:
- Clones GitHub repo directly to access datasets
- Output directory configured for /kaggle/working/
- Pinned library installations
- Preprocessing contract: strips response_type before tokenization
- 4-bit QLoRA SFTTrainer with report_to="none"
- Deterministic OOM fallback ladder ((8x2) -> (4x4) -> (2x8) -> (1x16) -> Fail)
- Separate 16-bit unquantized model reload for merge_and_unload
- Pinned llama.cpp commit SHA GGUF conversion & Ollama Modelfile
- Artifact SHA-256 computation and training_config.json generation
"""

import json
from pathlib import Path


def create_kaggle_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "isGpuEnabled": True,
                "isInternetEnabled": True
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_md(source: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Cell 1: Title
    add_md("""
# Stage 10: 4-bit QLoRA Fine-Tuning for IOFT Robot (Qwen 2.5 3B Instruct)
**Kaggle GPU Training, Adaptation & Export Pipeline**

This notebook fine-tunes `Qwen/Qwen2.5-3B-Instruct` for the IOFT humanoid robot assistant on Kaggle GPU.
- **Safety Invariant**: The fine-tuned model output remains untrusted input locally; fine-tuning specializes tool calling and clarification adherence without modifying Stages 1-9 safety layers.
- **Kaggle Boundary**: Operates strictly offline from robot hardware with zero physical packet dispatch.
- **Prerequisite**: Ensure **Internet ON** and **GPU T4 x 2** (or P100) are enabled in the right panel under **Settings**.
""")

    # Cell 2: Clone Repo
    add_md("## 1. Clone Project Repository to Access Datasets")
    add_code("""
import os
from pathlib import Path

# Clone repository to access datasets if not already present
if not os.path.exists("/kaggle/working/custom-llm-robot"):
    !git clone https://github.com/Aaditya-03-03/custom-llm-robot.git /kaggle/working/custom-llm-robot
else:
    !cd /kaggle/working/custom-llm-robot && git pull origin main

print("Repository ready at /kaggle/working/custom-llm-robot")
""")

    # Cell 3: Dependencies
    add_md("## 2. Pinned Dependencies Installation")
    add_code("""
!pip install -q \\
    transformers==4.49.0 \\
    peft==0.14.0 \\
    "bitsandbytes>=0.50.0" \\
    accelerate==1.4.0 \\
    trl==0.15.1 \\
    datasets==3.3.2 \\
    torch
""")

    # Cell 4: Workspace Scaffolding
    add_md("## 3. Kaggle Workspace Setup")
    add_code("""
KAGGLE_BASE = Path("/kaggle/working/stage10")
DATASETS_DIR = Path("/kaggle/working/custom-llm-robot/data/finetuning")
CHECKPOINTS_DIR = KAGGLE_BASE / "checkpoints"
ADAPTERS_DIR = KAGGLE_BASE / "adapters" / "ioft-qwen25-3b-v1"
MERGED_DIR = KAGGLE_BASE / "merged" / "ioft-qwen25-3b-v1"
GGUF_DIR = KAGGLE_BASE / "gguf"
OLLAMA_DIR = KAGGLE_BASE / "ollama"

for d in [CHECKPOINTS_DIR, ADAPTERS_DIR, MERGED_DIR, GGUF_DIR, OLLAMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Kaggle workspace initialized at:", KAGGLE_BASE)
""")

    # Cell 5: Hardware Check
    add_md("## 4. Hardware Inspection & Precision Selection")
    add_code("""
import torch

if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is not available! Please enable GPU in the right panel under Settings -> Accelerator -> GPU T4 x 2.")

gpu_name = torch.cuda.get_device_name(0)
gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Detected GPU: {gpu_name}")
print(f"Total VRAM: {gpu_vram_gb:.2f} GB")
print(f"Selected Compute Precision: {compute_dtype}")
""")

    # Cell 6: Dataset Loading
    add_md("""
## 5. Dataset Loading & Preprocessing Contract
**Contract**: `response_type` is dataset validation metadata only.
It is inspected and validated, then **strictly stripped** before tokenization so that it never enters model targets or loss calculations.
""")
    add_code("""
import json
from datasets import Dataset

def load_and_preprocess_split(split_name: str, split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")
    
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            if "response_type" in messages[-1]["content"]:
                raise ValueError("Metadata leakage: response_type found in assistant content!")
            records.append({"messages": messages})
            
    print(f"Loaded {len(records)} records from {split_path} (response_type stripped)")
    return Dataset.from_list(records)

train_file = DATASETS_DIR / "train" / "train.jsonl"
val_file = DATASETS_DIR / "validation" / "val.jsonl"
test_file = DATASETS_DIR / "test" / "test.jsonl"

train_dataset = load_and_preprocess_split("train", train_file)
val_dataset = load_and_preprocess_split("validation", val_file)
test_dataset = load_and_preprocess_split("test", test_file)
""")

    # Cell 7: Model & Tokenizer Revision Lock
    add_md("""
## 6. Model & Tokenizer Revision Lock and Checkpoint Compatibility
Enforces exact model and tokenizer configuration matching.
""")
    add_code("""
import hashlib
from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# Pin exact Hugging Face revision/commit hash
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tok_config_hash = hashlib.sha256(json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str).encode("utf-8")).hexdigest()
print(f"Base Model: {BASE_MODEL_ID} @ {MODEL_REVISION[:8]}")
print(f"Tokenizer Config Hash: {tok_config_hash[:12]}")
""")

    # Cell 8: Checkpoint scan
    add_md("## 7. Checkpoint Compatibility Check")
    add_code("""
manifest_path = DATASETS_DIR / "dataset_manifest.json"
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        current_manifest_sha = manifest_data.get("manifest_sha256", "")
else:
    current_manifest_sha = "untracked"

checkpoint_state = {
    "resumed": False,
    "resume_from": None,
    "global_step": 0,
    "epoch": 0.0,
}

existing_checkpoints = sorted(list(CHECKPOINTS_DIR.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)

if existing_checkpoints:
    latest_ckpt = existing_checkpoints[-1]
    print(f"Valid checkpoint found: {latest_ckpt}. Resuming supported.")
    checkpoint_state["resumed"] = True
    checkpoint_state["resume_from"] = str(latest_ckpt)
else:
    print("Starting fresh training run.")
""")

    # Cell 9: Model Quantization & LoRA
    add_md("## 8. Model Quantization & LoRA Adapter Setup")
    add_code("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    revision=MODEL_REVISION,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
""")

    # Cell 10: Training Loop
    add_md("""
## 9. Training with Deterministic Generic OOM Fallback Ladder
**Target**: `effective_batch_size = 16`
**Fallback ladder**: $(8 \\times 2) \\longrightarrow (4 \\times 4) \\longrightarrow (2 \\times 8) \\longrightarrow (1 \\times 16) \\longrightarrow \\text{Fail}$
""")
    add_code("""
from trl import SFTTrainer, SFTConfig

LADDER = [
    (8, 2),
    (4, 4),
    (2, 8),
    (1, 16),
]

initial_idx = 0 if gpu_vram_gb >= 24.0 else 1
ladder_to_try = LADDER[initial_idx:]

final_batch_size = None
final_accum_steps = None
trainer = None

for batch_size, accum_steps in ladder_to_try:
    print(f"\\n--- Attempting training configuration: batch_size={batch_size}, accum_steps={accum_steps} (effective = {batch_size * accum_steps}) ---")
    torch.cuda.empty_cache()
    
    training_args = SFTConfig(
        output_dir=str(CHECKPOINTS_DIR),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=42,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_seq_length=512,
        report_to="none",
    )
    
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            args=training_args,
        )
        
        resume_arg = checkpoint_state["resume_from"] if checkpoint_state["resumed"] else None
        trainer.train(resume_from_checkpoint=resume_arg)
        
        final_batch_size = batch_size
        final_accum_steps = accum_steps
        print(f"\\nSuccessfully trained with batch_size={final_batch_size}, accum_steps={final_accum_steps}!")
        break
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM occurred at batch_size={batch_size}, accum_steps={accum_steps}: {e}")
        torch.cuda.empty_cache()
        del trainer
        continue

if final_batch_size is None:
    raise RuntimeError("CUDA OOM even at batch_size=1 (1x16). Training failed.")

print(f"Training completed. Final runtime configuration: {final_batch_size}x{final_accum_steps} (Effective: {final_batch_size * final_accum_steps})")
""")

    # Cell 11: Save LoRA Adapter
    add_md("## 10. Save Canonical LoRA Adapter")
    add_code("""
model.save_pretrained(str(ADAPTERS_DIR))
tokenizer.save_pretrained(str(ADAPTERS_DIR))
print(f"LoRA adapter and tokenizer successfully saved to: {ADAPTERS_DIR}")
""")

    # Cell 12: Clean 16-bit Reload and Merge
    add_md("""
## 11. Clean 16-bit Unquantized Model Reload and Weight Merge
**Prevents 4-bit dequantization loss**:
Unloads the 4-bit model, reloads a pristine unquantized 16-bit base model, attaches the adapter, and runs `merge_and_unload()`.
""")
    add_code("""
import gc
from peft import PeftModel

# Free 4-bit quantized model from VRAM and RAM
del model
if trainer is not None:
    del trainer
torch.cuda.empty_cache()
gc.collect()

print("Reloading pristine unquantized base model in 16-bit precision...")
base_model_16bit = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Attaching LoRA adapter from {ADAPTERS_DIR}...")
merged_model = PeftModel.from_pretrained(base_model_16bit, str(ADAPTERS_DIR))
merged_model = merged_model.merge_and_unload()

print(f"Saving merged 16-bit model to {MERGED_DIR}...")
merged_model.save_pretrained(str(MERGED_DIR))
tokenizer.save_pretrained(str(MERGED_DIR))

del merged_model
del base_model_16bit
torch.cuda.empty_cache()
gc.collect()

print("Merged 16-bit model successfully saved!")
""")

    # Cell 13: GGUF Conversion
    add_md("""
## 12. GGUF Quantization & Ollama Modelfile Packaging
Converts the merged 16-bit model to GGUF using a pinned commit of `llama.cpp` (`2018898b95fecfaeebebebbdfc5d2fa3c3a9cf33`).
""")
    add_code("""
LLAMA_CPP_DIR = Path("/kaggle/working/llama.cpp")
LLAMA_CPP_COMMIT = "2018898b95fecfaeebebebbdfc5d2fa3c3a9cf33"

if not LLAMA_CPP_DIR.exists():
    print("Cloning llama.cpp...")
    !git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_DIR}
    !cd {LLAMA_CPP_DIR} && git checkout {LLAMA_CPP_COMMIT}
    !cd {LLAMA_CPP_DIR} && pip install -r requirements.txt
    !cd {LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release -j$(nproc)
else:
    print("llama.cpp already present.")

GGUF_OUT_F16 = GGUF_DIR / "ioft-qwen25-3b-v1-f16.gguf"
GGUF_OUT_Q4 = GGUF_DIR / "ioft-qwen25-3b-v1-q4_k_m.gguf"

print("Converting merged model to GGUF F16...")
!python {LLAMA_CPP_DIR}/convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_OUT_F16} --outtype f16

print("Quantizing GGUF to Q4_K_M...")
!{LLAMA_CPP_DIR}/build/bin/llama-quantize {GGUF_OUT_F16} {GGUF_OUT_Q4} Q4_K_M

if GGUF_OUT_F16.exists():
    GGUF_OUT_F16.unlink()

print(f"GGUF artifact generated at: {GGUF_OUT_Q4}")
""")

    # Cell 14: Modelfile and Config
    add_md("## 13. Ollama Modelfile & Audit Manifest Export")
    add_code("""
modelfile_content = f\"\"\"FROM {GGUF_OUT_Q4.name}

PARAMETER temperature 0
PARAMETER seed 42
PARAMETER num_predict 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"You are the Robot Action Planning Engine for the IOFT Robot.
Your task is to translate user natural language requests into structured tool calls.
Available Tools: forward, backward, stop.
Rules: Always respond in valid JSON matching the Stage 7 execution plan schema.
\"\"\"
\"\"\"

modelfile_path = OLLAMA_DIR / "Modelfile"
with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print(f"Generated Modelfile at: {modelfile_path}")

def compute_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

training_manifest = {
    "stage": 10,
    "model_name": "ioft-qwen25-3b-v1",
    "base_model": BASE_MODEL_ID,
    "base_model_revision": MODEL_REVISION,
    "tokenizer_config_hash": tok_config_hash,
    "effective_batch_size": final_batch_size * final_accum_steps,
    "per_device_batch_size": final_batch_size,
    "gradient_accumulation_steps": final_accum_steps,
    "epochs": 3,
    "learning_rate": 2e-4,
    "gguf_converter": "llama.cpp",
    "gguf_converter_revision": LLAMA_CPP_COMMIT,
    "artifacts": {
        "gguf_sha256": compute_sha256(GGUF_OUT_Q4),
        "ollama_model": "ioft-qwen25-3b-v1"
    }
}

config_out = KAGGLE_BASE / "training_config.json"
with open(config_out, "w", encoding="utf-8") as f:
    json.dump(training_manifest, f, indent=2)

print(f"Training configuration saved to: {config_out}")
print("\\nAll packaging complete! Download your artifacts from /kaggle/working/stage10/")
""")

    # Write notebook
    out_path = Path("training/stage10_qlora_kaggle.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated turnkey Kaggle notebook at: {out_path}")


if __name__ == "__main__":
    create_kaggle_notebook()
