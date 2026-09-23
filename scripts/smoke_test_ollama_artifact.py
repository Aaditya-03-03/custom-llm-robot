"""
Ollama Artifact Smoke Test (Stage 10).
Verifies the deployed local Ollama artifact prior to enabling USE_FINE_TUNED_MODEL=True:
1. Verifies GGUF file SHA-256 matches `artifacts.gguf_sha256` in training_config.json.
2. Asserts local Ollama daemon has the model imported (via 'ollama list' or /api/tags).
3. Executes deterministic inference probe with native Ollama options:
   - temperature = 0
   - seed = 42
   - num_predict = 256 (Ollama equivalent of max_new_tokens = 256)
4. Asserts that the response is valid Stage 7 JSON conforming to ExecutionPlan.
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.planning.parser import parse_execution_plan
from app.core.config import settings


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_gguf_integrity(config_path: Path, gguf_path: Path) -> bool:
    print(f"Checking GGUF artifact integrity: {gguf_path}...")
    if not gguf_path.exists():
        print(f"Warning: GGUF artifact not found at {gguf_path}.")
        return False

    if not config_path.exists():
        print(f"Warning: training_config.json not found at {config_path}.")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    expected_sha = cfg.get("artifacts", {}).get("gguf_sha256")
    if not expected_sha:
        print("Warning: 'artifacts.gguf_sha256' not found in training_config.json.")
        return False

    actual_sha = compute_sha256(gguf_path)
    if actual_sha != expected_sha:
        print(f"FAIL: GGUF SHA-256 mismatch! Expected {expected_sha}, got {actual_sha}")
        return False

    print(f"PASS: GGUF SHA-256 verified ({actual_sha[:16]}...)")
    return True


def check_ollama_model_exists(base_url: str, model_name: str) -> bool:
    url = f"{base_url.rstrip('/')}/api/tags"
    print(f"Querying Ollama tags at {url} for '{model_name}'...")
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in tags]
        target_stem = model_name.split(":")[0]
        if target_stem in model_names or model_name in [m.get("name", "") for m in tags]:
            print(f"PASS: Model '{model_name}' found in Ollama.")
            return True
        else:
            print(f"FAIL: Model '{model_name}' not found in Ollama tags: {model_names}")
            return False
    except Exception as e:
        print(f"Error querying Ollama daemon: {e}")
        return False


def run_deterministic_smoke_probe(base_url: str, model_name: str) -> bool:
    url = f"{base_url.rstrip('/')}/api/chat"
    prompt = "Move forward slowly for 2 steps"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 256,
        },
    }

    print(f"Dispatching deterministic smoke probe to {url} with options: {payload['options']}...")
    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        resp_data = resp.json()
        raw_content = resp_data.get("message", {}).get("content", "")
        print(f"Received raw response:\n{raw_content}")

        plan, err = parse_execution_plan(raw_content)
        if err:
            print(f"FAIL: Response could not be parsed into Stage 7 ExecutionPlan: {err}")
            return False

        if not plan.actions:
            print("FAIL: Parsed ExecutionPlan contains 0 actions for movement command!")
            return False

        first_act = plan.actions[0]
        if first_act.tool != "forward":
            print(f"FAIL: Expected tool 'forward', got '{first_act.tool}'")
            return False

        print(f"PASS: Valid Stage 7 ExecutionPlan generated (Tool: {first_act.tool}, Params: {first_act.parameters})")
        return True
    except Exception as e:
        print(f"Error executing smoke probe: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ollama Artifact Smoke Test for Stage 10")
    parser.add_argument("--config", type=Path, default=Path("data/finetuning/training_config.json"), help="Path to training_config.json")
    parser.add_argument("--gguf", type=Path, default=Path("data/finetuning/ioft-qwen25-3b-v1.gguf"), help="Path to GGUF file")
    parser.add_argument("--url", type=str, default=settings.OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--model", type=str, default=settings.LLM_FINE_TUNED_MODEL_NAME, help="Model name tag in Ollama")
    parser.add_argument("--skip-file-check", action="store_true", help="Skip local file SHA-256 check if testing remote daemon")

    args = parser.parse_args()

    print("=== Stage 10 Ollama Artifact Smoke Test ===")
    all_passed = True

    if not args.skip_file_check:
        if args.gguf.exists():
            file_ok = verify_gguf_integrity(args.config, args.gguf)
            if not file_ok:
                all_passed = False
        else:
            print(f"Note: Local GGUF file not present at {args.gguf}; skipping local hash check.")

    exists_ok = check_ollama_model_exists(args.url, args.model)
    if not exists_ok:
        print("\nSmoke test incomplete: Ollama daemon is not running or model is not yet imported.")
        print("To import model into Ollama:")
        print(f"  ollama create {args.model} -f /path/to/Modelfile")
        sys.exit(1)

    probe_ok = run_deterministic_smoke_probe(args.url, args.model)
    if not probe_ok:
        all_passed = False

    if all_passed:
        print("\n=== SMOKE TEST PASSED: Model is ready for local deployment! ===")
        sys.exit(0)
    else:
        print("\n=== SMOKE TEST FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
