"""
Response-Type Aware Dataset Validator and Security Sanitizer (Stage 10).
Validates:
1. Response-type aware schema checks:
   - ACTION_PLAN: tool != null, in ToolRegistry, valid speed/profile and steps
   - CLARIFICATION: tool != null, parameters == {}, non-empty explanation
   - UNSUPPORTED: tool == null, reason non-empty
   - CONVERSATIONAL: actions == [], plan_explanation non-empty
2. Invariant: response_type is top-level metadata only; never inside assistant JSON.
3. Security: No raw UDP commands (e.g. FORWARD|...), no secrets or keys.
4. Leakage & Duplication:
   - Within-split duplicate detection
   - Cross-split exact match detection (Error)
   - Cross-split token Jaccard similarity > 0.85 (Error)
5. Dataset Manifest: Computes and writes SHA-256 hashes to data/finetuning/dataset_manifest.json.
"""

import sys
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tools.registry import default_tool_registry
from app.safety.validator import CommandSafetyValidator

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about",
    "and", "or", "then", "of", "is", "am", "are", "you", "me", "my", "your",
    "robot", "assistant", "ioft", "please", "could", "would", "can", "hey",
}

RAW_UDP_PATTERNS = [
    re.compile(r"\bFORWARD\|\d+", re.IGNORECASE),
    re.compile(r"\bBACKWARD\|\d+", re.IGNORECASE),
    re.compile(r"\bSTOP\b"),
]

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{8,}['\"]"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
]


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def tokenize_query(query: str) -> Set[str]:
    """Tokenize user query into non-stopword tokens for Jaccard similarity."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
    return {t for t in tokens if t not in STOPWORDS}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def validate_record(line_idx: int, record: Dict[str, Any], split: str) -> List[str]:
    """Validate a single record according to its response_type and security rules."""
    errors = []

    # 1. Wrapper structure
    if "response_type" not in record:
        errors.append(f"Line {line_idx}: Missing 'response_type' metadata.")
        return errors

    resp_type = record["response_type"]
    if resp_type not in ("ACTION_PLAN", "CLARIFICATION", "UNSUPPORTED", "CONVERSATIONAL"):
        errors.append(f"Line {line_idx}: Invalid response_type '{resp_type}'.")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"Line {line_idx}: 'messages' must be a list of 3 items (system, user, assistant).")
        return errors

    system_msg, user_msg, assistant_msg = messages[0], messages[1], messages[2]
    user_content = user_msg.get("content", "")
    assistant_content = assistant_msg.get("content", "")

    # 2. Security sanitization
    for pat in RAW_UDP_PATTERNS:
        if pat.search(user_content) or pat.search(assistant_content):
            errors.append(f"Line {line_idx}: Raw UDP protocol command leaked into content!")
    for pat in SECRET_PATTERNS:
        if pat.search(user_content) or pat.search(assistant_content):
            errors.append(f"Line {line_idx}: Sensitive credential or API key pattern detected!")

    # 3. response_type leakage check: response_type must NEVER enter assistant content
    if "response_type" in assistant_content:
        errors.append(f"Line {line_idx}: 'response_type' leaked into assistant JSON content!")

    # 4. Assistant JSON parsing
    try:
        assistant_json = json.loads(assistant_content)
    except json.JSONDecodeError as e:
        errors.append(f"Line {line_idx}: Assistant content is not valid JSON: {e}")
        return errors

    explanation = assistant_json.get("plan_explanation")
    actions = assistant_json.get("actions")

    if not isinstance(explanation, str) or not explanation.strip():
        errors.append(f"Line {line_idx}: Missing or empty 'plan_explanation'.")
    if not isinstance(actions, list):
        errors.append(f"Line {line_idx}: 'actions' must be a list.")
        return errors

    # 5. Type-specific validations
    if resp_type == "CONVERSATIONAL":
        if len(actions) != 0:
            errors.append(f"Line {line_idx}: CONVERSATIONAL record must have empty actions list, got {len(actions)}.")

    elif resp_type == "UNSUPPORTED":
        if len(actions) != 1:
            errors.append(f"Line {line_idx}: UNSUPPORTED record must have exactly 1 action, got {len(actions)}.")
        else:
            action = actions[0]
            if action.get("tool") is not None:
                errors.append(f"Line {line_idx}: UNSUPPORTED record action.tool must be null/None, got '{action.get('tool')}'.")
            reason = action.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"Line {line_idx}: UNSUPPORTED record must have non-empty 'reason'.")

    elif resp_type == "CLARIFICATION":
        if len(actions) != 1:
            errors.append(f"Line {line_idx}: CLARIFICATION record must have exactly 1 action, got {len(actions)}.")
        else:
            action = actions[0]
            tool = action.get("tool")
            if not tool or not default_tool_registry.is_registered(tool):
                errors.append(f"Line {line_idx}: CLARIFICATION record must target a registered tool, got '{tool}'.")
            params = action.get("parameters")
            if params != {}:
                errors.append(f"Line {line_idx}: CLARIFICATION action parameters must be empty dict {{}}, got {params}.")

    elif resp_type == "ACTION_PLAN":
        if len(actions) == 0:
            errors.append(f"Line {line_idx}: ACTION_PLAN must have at least 1 action.")
        elif len(actions) > 3:
            errors.append(f"Line {line_idx}: ACTION_PLAN exceeds max actions (3), got {len(actions)}.")

        for a_idx, action in enumerate(actions, start=1):
            tool = action.get("tool")
            if not tool:
                errors.append(f"Line {line_idx}, action {a_idx}: Tool cannot be null for ACTION_PLAN.")
                continue
            if not default_tool_registry.is_registered(tool):
                errors.append(f"Line {line_idx}, action {a_idx}: Tool '{tool}' is not registered in ToolRegistry.")
                continue

            params = action.get("parameters")
            if not isinstance(params, dict):
                errors.append(f"Line {line_idx}, action {a_idx}: Parameters must be a dict.")
                continue

            if tool == "stop":
                if params and any(v is not None for v in params.values()):
                    errors.append(f"Line {line_idx}, action {a_idx}: Tool 'stop' must have empty parameters, got {params}.")
            elif tool in ("forward", "backward"):
                speed = params.get("speed")
                profile = params.get("speed_profile")
                steps = params.get("steps", 1)

                if speed is None and profile is None:
                    errors.append(f"Line {line_idx}, action {a_idx}: Must provide either 'speed' or 'speed_profile'.")
                if speed is not None and profile is not None:
                    errors.append(f"Line {line_idx}, action {a_idx}: Cannot provide both 'speed' and 'speed_profile'.")
                if speed is not None and (not isinstance(speed, int) or speed < 1 or speed > 100):
                    errors.append(f"Line {line_idx}, action {a_idx}: Speed must be int between 1 and 100, got {speed}.")
                if profile is not None and profile not in ("slow", "medium", "fast"):
                    errors.append(f"Line {line_idx}, action {a_idx}: Profile must be slow/medium/fast, got {profile}.")
                if steps is not None and (not isinstance(steps, int) or steps < 1 or steps > 10):
                    errors.append(f"Line {line_idx}, action {a_idx}: Steps must be int between 1 and 10, got {steps}.")

    return errors


def main():
    print("Validating Stage 10 Fine-Tuning Datasets...")
    dataset_dir = Path("data/finetuning")
    splits = ["train", "val", "test"]
    split_files = {
        "train": dataset_dir / "train" / "train.jsonl",
        "val": dataset_dir / "validation" / "val.jsonl",
        "test": dataset_dir / "test" / "test.jsonl",
    }

    all_errors = []
    split_queries: Dict[str, List[Tuple[str, Set[str]]]] = {"train": [], "val": [], "test": []}
    split_hashes = {}

    for split in splits:
        path = split_files[split]
        if not path.exists():
            print(f"Error: Dataset split file not found: {path}")
            sys.exit(1)

        split_hashes[f"{split}_sha256"] = compute_sha256(path)
        print(f"[{split.upper()}] SHA-256: {split_hashes[f'{split}_sha256']}")

        seen_queries_in_split = set()
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    all_errors.append(f"{split} line {idx}: Invalid JSON: {e}")
                    continue

                rec_errors = validate_record(idx, record, split)
                if rec_errors:
                    all_errors.extend([f"[{split.upper()}] {err}" for err in rec_errors])

                # Extract user query
                user_msg = record.get("messages", [{}, {}])[1].get("content", "")
                if user_msg:
                    if user_msg in seen_queries_in_split:
                        all_errors.append(f"[{split.upper()}] Line {idx}: Exact duplicate query in split: '{user_msg}'")
                    seen_queries_in_split.add(user_msg)
                    split_queries[split].append((user_msg, tokenize_query(user_msg)))

    print(f"Total records validated: Train={len(split_queries['train'])}, Val={len(split_queries['val'])}, Test={len(split_queries['test'])}")

    # Cross-split leakage checks
    print("Checking cross-split leakage (Exact match & Jaccard similarity > 0.85)...")
    train_queries = split_queries["train"]
    for eval_split in ["val", "test"]:
        eval_queries = split_queries[eval_split]
        for e_q, e_tokens in eval_queries:
            for t_q, t_tokens in train_queries:
                # 1. Exact match cross split
                if e_q.lower().strip() == t_q.lower().strip():
                    all_errors.append(f"Cross-split exact leakage: '{e_q}' appears in both TRAIN and {eval_split.upper()}")
                # 2. Jaccard similarity check
                sim = jaccard_similarity(e_tokens, t_tokens)
                if sim > 0.85:
                    all_errors.append(f"Cross-split Jaccard leakage ({sim:.2f} > 0.85): '{e_q}' ({eval_split.upper()}) vs '{t_q}' (TRAIN)")

    if all_errors:
        print(f"\nVALIDATION FAILED with {len(all_errors)} errors:")
        for err in all_errors[:20]:
            print(f"  - {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors.")
        sys.exit(1)

    # Combined manifest hash
    combined_hasher = hashlib.sha256()
    for s in ["train", "val", "test"]:
        combined_hasher.update(split_hashes[f"{s}_sha256"].encode("utf-8"))
    split_hashes["manifest_sha256"] = combined_hasher.hexdigest()

    manifest_path = dataset_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_hashes, f, indent=2)

    print(f"\nVALIDATION PASSED with 0 errors!")
    print(f"Manifest written to {manifest_path}:")
    print(json.dumps(split_hashes, indent=2))


if __name__ == "__main__":
    main()
