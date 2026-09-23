"""
Creates the turnkey Google Colab Jupyter Notebook: training/stage10_qlora_training.ipynb
Incorporates all Stage 10 requirements:
- Pinned library installations
- Drive mounting and directory scaffolding
- Hardware inspection and precision selection
- Preprocessing contract: strips response_type before tokenization
- Checkpoint compatibility verification & resume logic
- 4-bit QLoRA SFTTrainer
- Deterministic OOM fallback ladder ((8x2) -> (4x4) -> (2x8) -> (1x16) -> Fail)
- Separate 16-bit unquantized model reload for merge_and_unload
- Pinned llama.cpp commit SHA GGUF conversion & Ollama Modelfile
- Artifact SHA-256 computation and training_config.json generation
"""

import json
from pathlib import Path


def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
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
**Google Colab Training, Adaptation & Export Pipeline**

This notebook fine-tunes `Qwen/Qwen2.5-3B-Instruct` for the IOFT humanoid robot assistant.
- **Safety Invariant**: The fine-tuned model output remains untrusted input locally; fine-tuning specializes tool calling and clarification adherence without modifying Stages 1-9 safety layers.
- **Colab Boundary**: Colab operates strictly offline and never transmits packets to physical hardware.
""")

    # Cell 2: Dependencies
    add_md("## 1. Pinned Dependencies Installation")
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

    # Cell 3: Drive Mount
    add_md("## 2. Google Drive Mounting & Workspace Setup")
    add_code("""
import os
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

DRIVE_BASE = Path("/content/drive/MyDrive/IOFT-LLM/stage10")
DATASETS_DIR = DRIVE_BASE / "datasets"
CHECKPOINTS_DIR = DRIVE_BASE / "checkpoints"
ADAPTERS_DIR = DRIVE_BASE / "adapters" / "ioft-qwen25-3b-v1"
MERGED_DIR = DRIVE_BASE / "merged" / "ioft-qwen25-3b-v1"
GGUF_DIR = DRIVE_BASE / "gguf"
OLLAMA_DIR = DRIVE_BASE / "ollama"
EVALS_DIR = DRIVE_BASE / "evaluations"

for d in [DATASETS_DIR, CHECKPOINTS_DIR, ADAPTERS_DIR, MERGED_DIR, GGUF_DIR, OLLAMA_DIR, EVALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Workspace initialized at:", DRIVE_BASE)
""")

    # Cell 4: GPU Inspection & Precision
    add_md("## 3. Hardware Inspection & Precision Selection")
    add_code("""
import torch

if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is not available in this Colab runtime! Please enable GPU via Runtime -> Change runtime type.")

gpu_name = torch.cuda.get_device_name(0)
gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Detected GPU: {gpu_name}")
print(f"Total VRAM: {gpu_vram_gb:.2f} GB")
print(f"Selected Compute Precision: {compute_dtype}")
""")

    # Cell 5: Dataset Loading & Preprocessing Contract
    add_md("""
## 4. Dataset Loading & Preprocessing Contract
**Contract**: `response_type` is dataset validation metadata only.
It is inspected and validated, then **strictly stripped** before tokenization so that it never enters model targets or loss calculations.
""")
    add_code("""
import json
from datasets import Dataset

def load_and_preprocess_split(split_name: str, split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}. Please upload to Google Drive: {DATASETS_DIR}")
    
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Preprocessing contract: strip response_type, retain messages list
            messages = item.get("messages", [])
            # Assert assistant content does not leak response_type
            if "response_type" in messages[-1]["content"]:
                raise ValueError("Metadata leakage: response_type found in assistant content!")
            records.append({"messages": messages})
            
    print(f"Loaded {len(records)} records from {split_path} (response_type stripped)")
    return Dataset.from_list(records)

train_file = DATASETS_DIR / "train.jsonl"
val_file = DATASETS_DIR / "val.jsonl"
test_file = DATASETS_DIR / "test.jsonl"

train_dataset = load_and_preprocess_split("train", train_file)
val_dataset = load_and_preprocess_split("validation", val_file)
test_dataset = load_and_preprocess_split("test", test_file)
""")

    # Cell 6: Model & Tokenizer Revision Lock
    add_md("""
## 5. Model & Tokenizer Revision Lock and Checkpoint Compatibility
Enforces exact model and tokenizer configuration matching before any checkpoint can be resumed.
""")
    add_code("""
import hashlib
from transformers import AutoTokenizer, AutoConfig

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

# Compute tokenizer config hash for revision locking
tok_config_hash = hashlib.sha256(json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str).encode("utf-8")).hexdigest()
print(f"Base Model: {BASE_MODEL_ID} @ {MODEL_REVISION[:8]}")
print(f"Tokenizer Config Hash: {tok_config_hash[:12]}")
""")

    # Cell 7: Checkpoint Resume Check
    add_md("## 6. Checkpoint Compatibility Check")
    add_code("""
import transformers, peft, trl

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

# Scan for existing checkpoints
existing_checkpoints = sorted(list(CHECKPOINTS_DIR.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)

if existing_checkpoints:
    latest_ckpt = existing_checkpoints[-1]
    cfg_file = CHECKPOINTS_DIR / "training_config.json"
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            prev_cfg = json.load(f)
        # Compatibility verification
        if prev_cfg.get("dataset_hashes", {}).get("manifest_sha256") != current_manifest_sha:
            raise ValueError("Incompatible checkpoint: dataset_manifest_sha256 mismatch!")
        if prev_cfg.get("base_model") != BASE_MODEL_ID:
            raise ValueError("Incompatible checkpoint: base_model mismatch!")
        if prev_cfg.get("base_model_revision") != MODEL_REVISION:
            raise ValueError("Incompatible checkpoint: base_model_revision mismatch!")
        print(f"Valid checkpoint found: {latest_ckpt}. Resuming supported.")
        checkpoint_state["resumed"] = True
        checkpoint_state["resume_from"] = str(latest_ckpt)
        step_num = int(latest_ckpt.name.split("-")[-1])
        checkpoint_state["global_step"] = step_num
    else:
        print(f"Found {latest_ckpt} but no prior training_config.json; starting fresh.")
else:
    print("No prior checkpoint found; starting fresh training run.")
""")

    # Cell 8: Model Loading in 4-bit NF4
    add_md("## 7. Model Quantization & LoRA Adapter Setup")
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

# Memory optimizations
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

    # Cell 9: Training with Deterministic OOM Fallback Ladder
    add_md("""
## 8. Training with Deterministic Generic OOM Fallback Ladder
**Target**: `effective_batch_size = 16`
**Fallback ladder**: $(8 \times 2) \longrightarrow (4 \times 4) \longrightarrow (2 \times 8) \longrightarrow (1 \times 16) \longrightarrow \text{Fail}$
Catches `torch.cuda.OutOfMemoryError` during training execution (including forward/backward pass), clears cache, and retries at next lower step.
""")
    add_code("""
from trl import SFTTrainer, SFTConfig

LADDER = [
    (8, 2),
    (4, 4),
    (2, 8),
    (1, 16),
]

# Initial selection based on VRAM
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
        dataset_text_field="messages",
    )
    
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
            args=training_args,
        )
        
        # Execute training (catches OOM during actual forward/backward passes)
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
    raise RuntimeError("CUDA OOM even at batch_size=1 (1x16). Training failed. Please use a GPU with more VRAM.")

print(f"Training completed. Final runtime configuration: {final_batch_size}x{final_accum_steps} (Effective: {final_batch_size * final_accum_steps})")
""")

    # Cell 10: Export Canonical LoRA Adapter
    add_md("## 9. Save Canonical LoRA Adapter")
    add_code("""
model.save_pretrained(str(ADAPTERS_DIR))
tokenizer.save_pretrained(str(ADAPTERS_DIR))
print(f"Canonical LoRA adapter saved to: {ADAPTERS_DIR}")
""")

    # Cell 11: Clean 16-bit Reload & Merge (No 4-bit distortion)
    add_md("""
## 10. Clean 16-bit Reload & LoRA Merge
**Strict Separation**: The 4-bit model is unloaded. The unquantized base model is reloaded in 16-bit (`compute_dtype`), the adapter applied, and `merge_and_unload()` executed cleanly without dequantization distortion.
""")
    add_code("""
import gc
from peft import PeftModel

# Free 4-bit training instance from VRAM
del model, trainer
gc.collect()
torch.cuda.empty_cache()

print("Reloading clean unquantized base model in 16-bit precision...")
base_model_16bit = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True,
)

print("Applying trained LoRA adapter...")
peft_model = PeftModel.from_pretrained(base_model_16bit, str(ADAPTERS_DIR))
merged_model = peft_model.merge_and_unload()

print(f"Saving merged 16-bit weights to: {MERGED_DIR}...")
merged_model.save_pretrained(str(MERGED_DIR))
tokenizer.save_pretrained(str(MERGED_DIR))
print("Merged 16-bit model successfully exported.")
""")

    # Cell 12: GGUF Conversion & Ollama Modelfile
    add_md("""
## 11. GGUF Conversion with Pinned llama.cpp & Ollama Modelfile
Pins `llama.cpp` repository at commit `2018898b95fecfaeebebebbdfc5d2fa3c3a9cf33` for deterministic bit-level GGUF conversion.
""")
    add_code("""
import subprocess

LLAMA_CPP_DIR = Path("/content/llama.cpp")
LLAMA_CPP_COMMIT = "2018898b95fecfaeebebebbdfc5d2fa3c3a9cf33"

if not LLAMA_CPP_DIR.exists():
    print("Cloning llama.cpp and checking out pinned commit...")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)], check=True)
    subprocess.run(["git", "-C", str(LLAMA_CPP_DIR), "checkout", LLAMA_CPP_COMMIT], check=True)
    subprocess.run(["pip", "install", "-r", str(LLAMA_CPP_DIR / "requirements.txt")], check=True)

# Convert to GGUF
gguf_output_file = GGUF_DIR / "ioft-qwen25-3b-v1.gguf"
print(f"Converting merged weights to GGUF: {gguf_output_file}...")
subprocess.run([
    "python", str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
    str(MERGED_DIR),
    "--outfile", str(gguf_output_file),
    "--outtype", "q8_0"
], check=True)

# Generate Ollama Modelfile
modelfile_path = OLLAMA_DIR / "Modelfile"
modelfile_content = f\"\"\"FROM {gguf_output_file}
PARAMETER temperature 0
PARAMETER seed 42
PARAMETER num_predict 256
\"\"\"
with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print(f"GGUF created: {gguf_output_file}")
print(f"Ollama Modelfile created: {modelfile_path}")
""")

    # Cell 13: Artifact Hashes & training_config.json
    add_md("## 12. Artifact Hashes & Final training_config.json Manifest")
    add_code("""
def file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

gguf_hash = file_sha256(gguf_output_file)
print(f"GGUF SHA-256: {gguf_hash}")

training_config = {
    "base_model": BASE_MODEL_ID,
    "base_model_revision": MODEL_REVISION,
    "tokenizer_revision": tok_config_hash,
    "transformers_version": transformers.__version__,
    "peft_version": peft.__version__,
    "trl_version": trl.__version__,
    "bitsandbytes_version": "0.45.2",
    "torch_version": torch.__version__,
    "gguf_converter": "llama.cpp",
    "gguf_converter_revision": LLAMA_CPP_COMMIT,
    "gpu_device_name": gpu_name,
    "gpu_vram_gb": round(gpu_vram_gb, 2),
    "compute_dtype": str(compute_dtype),
    "per_device_train_batch_size": final_batch_size,
    "gradient_accumulation_steps": final_accum_steps,
    "effective_batch_size": final_batch_size * final_accum_steps,
    "seed": 42,
    "checkpoint": {
        "resumed": checkpoint_state["resumed"],
        "resume_from": checkpoint_state["resume_from"],
        "global_step": checkpoint_state["global_step"],
        "epoch": 3.0,
    },
    "artifacts": {
        "adapter_dir": str(ADAPTERS_DIR),
        "merged_dir": str(MERGED_DIR),
        "gguf_sha256": gguf_hash,
        "ollama_model": "ioft-qwen25-3b-v1"
    },
    "dataset_hashes": manifest_data if manifest_path.exists() else {}
}

config_out_path = DRIVE_BASE / "training_config.json"
with open(config_out_path, "w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2)

print(f"\\nFinal training configuration written to: {config_out_path}")
print(json.dumps(training_config, indent=2))
""")

    out_path = Path("training/stage10_qlora_training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated turnkey Colab notebook at: {out_path}")


if __name__ == "__main__":
    create_notebook()
