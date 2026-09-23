"""
Fine-Tuning Dataset Generator for IOFT Robot (Stage 10).
Generates balanced, leak-free train, validation, and test datasets in ChatML JSONL format.
Enforces semantic group splitting across 8 core categories:
1. Normal Locomotion & Stop
2. Speed Profiles (slow, medium, fast)
3. Explicit Numeric Speeds
4. Step Counts
5. Missing Parameters / Clarification (parameters: {})
6. Multi-Action Sequences (strictly forward, backward, stop)
7. Unsupported / Invalid Tools (tool: null, reason: "...")
8. Conversational / Chit-Chat (actions: [])
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.tool_calling import TOOL_CALLING_SYSTEM_PROMPT
from app.tools.schemas import get_available_tool_schemas
from app.core.config import settings

# Build canonical system prompt
_tool_schemas = json.dumps(get_available_tool_schemas(), indent=2)
CANONICAL_SYSTEM_PROMPT = TOOL_CALLING_SYSTEM_PROMPT.format(
    tool_schemas=_tool_schemas,
    max_actions=settings.MAX_ACTIONS_PER_PLAN,
)

OUTPUT_DIR = Path("data/finetuning")
RANDOM_SEED = 42

WRAPPER_TEMPLATES = [
    "{cmd}",
    "please {cmd}",
    "robot, {cmd}",
    "could you {cmd}",
    "can you {cmd} please",
    "ioft, {cmd}",
    "hey robot, {cmd}",
    "assistant, please {cmd}",
    "i need you to {cmd}",
    "kindly {cmd} immediately",
    "would you {cmd}",
    "robot, please {cmd} now",
    "please go ahead and {cmd}",
    "{cmd} right now",
    "{cmd} thanks",
    "assistant, {cmd}",
    "hey ioft, {cmd}",
    "can the robot {cmd}",
    "please {cmd} right away",
    "robot, kindly {cmd}",
    "{cmd} if possible",
    "now {cmd}",
    "command: {cmd}",
    "instruction: {cmd}",
    "ioft robot, {cmd}",
]


def format_example(response_type: str, user_text: str, assistant_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Wraps ChatML messages with top-level response_type metadata."""
    return {
        "response_type": response_type,
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(assistant_obj, separators=(",", ":"))},
        ],
    }


def generate_category_1_locomotion() -> Dict[str, List[Dict[str, Any]]]:
    """Category 1: Normal Locomotion & Stop commands."""
    groups = {"train": [], "val": [], "test": []}

    train_templates = [
        ("move forward slowly", "forward", "slow", 1),
        ("move backward slowly", "backward", "slow", 1),
        ("walk forward at medium speed", "forward", "medium", 1),
        ("walk backward at medium speed", "backward", "medium", 1),
        ("stop", "stop", None, 0),
        ("stop moving immediately", "stop", None, 0),
        ("halt now", "stop", None, 0),
        ("halt all locomotion", "stop", None, 0),
        ("emergency stop", "stop", None, 0),
        ("freeze in place", "stop", None, 0),
        ("advance forward at slow speed", "forward", "slow", 1),
        ("step back slowly", "backward", "slow", 1),
    ]

    val_templates = [
        ("go forward slowly", "forward", "slow", 1),
        ("go backward at medium speed", "backward", "medium", 1),
        ("bring the robot to a complete halt", "stop", None, 0),
        ("cease movement", "stop", None, 0),
    ]

    test_templates = [
        ("proceed forward with slow pace", "forward", "slow", 1),
        ("reverse backward at medium rate", "backward", "medium", 1),
        ("terminate locomotion right now", "stop", None, 0),
        ("discontinue walking", "stop", None, 0),
    ]

    for split, templates in [("train", train_templates), ("val", val_templates), ("test", test_templates)]:
        multiplier = 20 if split == "train" else 15
        for user_cmd, tool, profile, steps in templates:
            for i in range(multiplier):
                variant_text = WRAPPER_TEMPLATES[i].format(cmd=user_cmd)
                if tool == "stop":
                    plan = {
                        "plan_explanation": f"Halt all robot locomotion ({variant_text})",
                        "actions": [{"action_id": "action_1", "tool": "stop", "parameters": {}}],
                    }
                else:
                    plan = {
                        "plan_explanation": f"Execute {tool} movement at {profile} speed",
                        "actions": [
                            {
                                "action_id": "action_1",
                                "tool": tool,
                                "parameters": {"speed_profile": profile, "steps": steps},
                            }
                        ],
                    }
                groups[split].append(format_example("ACTION_PLAN", variant_text, plan))

    return groups


def generate_category_2_speed_profiles() -> Dict[str, List[Dict[str, Any]]]:
    """Category 2: Speed profiles (slow, medium, fast)."""
    groups = {"train": [], "val": [], "test": []}

    train_families = [
        ("creep forward very slowly", "forward", "slow"),
        ("step forward at slow velocity", "forward", "slow"),
        ("move backward moderately", "backward", "medium"),
        ("back up at normal speed", "backward", "medium"),
        ("sprint forward quickly", "forward", "fast"),
        ("move forward as fast as possible", "forward", "fast"),
        ("retreat backward quickly", "backward", "fast"),
        ("travel backward with high speed", "backward", "fast"),
    ]

    val_families = [
        ("glide forward gently and slowly", "forward", "slow"),
        ("shift backward at regular pace", "backward", "medium"),
        ("dash forward rapidly", "forward", "fast"),
        ("backpedal at top speed", "backward", "fast"),
    ]

    test_families = [
        ("crawl forward cautiously", "forward", "slow"),
        ("step back with standard velocity", "backward", "medium"),
        ("rush forward at maximum speed", "forward", "fast"),
        ("hurry backward briskly", "backward", "fast"),
    ]

    for split, families in [("train", train_families), ("val", val_families), ("test", test_families)]:
        reps = 25 if split == "train" else 10
        for phrase, tool, profile in families:
            for r in range(reps):
                text = WRAPPER_TEMPLATES[r].format(cmd=phrase)
                plan = {
                    "plan_explanation": f"Move {tool} using {profile} speed profile",
                    "actions": [
                        {
                            "action_id": "action_1",
                            "tool": tool,
                            "parameters": {"speed_profile": profile, "steps": 1},
                        }
                    ],
                }
                groups[split].append(format_example("ACTION_PLAN", text, plan))

    return groups


def generate_category_3_numeric_speeds() -> Dict[str, List[Dict[str, Any]]]:
    """Category 3: Explicit numeric speeds (1-100)."""
    groups = {"train": [], "val": [], "test": []}

    train_speeds = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    val_speeds = [12, 28, 42, 68]
    test_speeds = [5, 22, 48, 62, 90]

    train_tmpls = [
        "move {tool} at speed {sp} for {st} steps",
        "proceed {tool} at {sp} percent speed for {st} steps",
        "drive {tool} at exact speed {sp} taking {st} steps",
        "travel {tool} with speed {sp} for {st} steps",
    ]
    val_tmpls = [
        "navigate {tool} at exact speed {sp} for {st} steps",
        "guide {tool} setting speed to {sp} for {st} steps",
    ]
    test_tmpls = [
        "propel {tool} at calibrated speed {sp} for {st} steps",
        "direct {tool} with velocity {sp} for {st} steps",
    ]

    for split, speeds, tmpls in [
        ("train", train_speeds, train_tmpls),
        ("val", val_speeds, val_tmpls),
        ("test", test_speeds, test_tmpls),
    ]:
        for sp in speeds:
            for tool in ["forward", "backward"]:
                for step_cnt in [1, 2]:
                    for tmpl in tmpls:
                        text = tmpl.format(tool=tool, sp=sp, st=step_cnt)
                        if step_cnt == 1:
                            text = text.replace("1 steps", "1 step")
                        plan = {
                            "plan_explanation": f"Move {tool} at exact speed {sp}",
                            "actions": [
                                {
                                    "action_id": "action_1",
                                    "tool": tool,
                                    "parameters": {"speed": sp, "steps": step_cnt},
                                }
                            ],
                        }
                        groups[split].append(format_example("ACTION_PLAN", text, plan))

    return groups


def generate_category_4_step_counts() -> Dict[str, List[Dict[str, Any]]]:
    """Category 4: Step Counts (1 to 10 steps)."""
    groups = {"train": [], "val": [], "test": []}

    train_steps = [2, 3, 4, 6, 7, 8]
    val_steps = [5, 9]
    test_steps = [1, 10]

    train_step_tmpls = [
        "take {st} steps {tool} slowly",
        "walk {st} steps {tool} at medium speed",
        "move {tool} for {st} steps at slow pace",
        "advance {tool} {st} steps at medium velocity",
    ]
    val_step_tmpls = [
        "traverse {st} steps {tool} at slow rate",
        "stride {st} steps {tool} with medium speed",
    ]
    test_step_tmpls = [
        "step {st} paces {tool} slowly",
        "pace {st} units {tool} at medium speed",
    ]

    for split, step_list, tmpls in [
        ("train", train_steps, train_step_tmpls),
        ("val", val_steps, val_step_tmpls),
        ("test", test_steps, test_step_tmpls),
    ]:
        for st in step_list:
            for tool in ["forward", "backward"]:
                for tmpl in tmpls:
                    prof = "slow" if "slow" in tmpl else "medium"
                    text = tmpl.format(st=st, tool=tool)
                    plan = {
                        "plan_explanation": f"Move {tool} for {st} steps with {prof} speed",
                        "actions": [
                            {
                                "action_id": "action_1",
                                "tool": tool,
                                "parameters": {"speed_profile": prof, "steps": st},
                            }
                        ],
                    }
                    groups[split].append(format_example("ACTION_PLAN", text, plan))

    return groups


def generate_category_5_clarification() -> Dict[str, List[Dict[str, Any]]]:
    """Category 5: Missing speed parameters requiring clarification (parameters: {})."""
    groups = {"train": [], "val": [], "test": []}

    train_queries = [
        "move forward",
        "move backward",
        "walk forward",
        "walk backward",
        "advance forward",
        "retreat backward",
        "head forward",
        "head backward",
        "step forward",
        "step backward",
    ]

    val_queries = [
        "proceed forward",
        "recede backward",
        "navigate forward",
        "navigate backward",
    ]

    test_queries = [
        "drive forward",
        "drive backward",
        "march forward",
        "march backward",
    ]

    for split, queries in [("train", train_queries), ("val", val_queries), ("test", test_queries)]:
        reps = 15 if split == "train" else 8
        for q in queries:
            tool = "backward" if "back" in q or "recede" in q or "retreat" in q else "forward"
            for i in range(reps):
                text = WRAPPER_TEMPLATES[i].format(cmd=q)
                plan = {
                    "plan_explanation": f"Missing speed for {tool} movement; clarification required",
                    "actions": [
                        {
                            "action_id": "action_1",
                            "tool": tool,
                            "parameters": {},
                        }
                    ],
                }
                groups[split].append(format_example("CLARIFICATION", text, plan))

    return groups


def generate_category_6_multi_action() -> Dict[str, List[Dict[str, Any]]]:
    """Category 6: Multi-Action sequences (strictly frozen Stage 7 tools)."""
    groups = {"train": [], "val": [], "test": []}

    train_seqs = [
        (
            "move forward slowly then backward slowly",
            [("forward", "slow", 1), ("backward", "slow", 1)],
            "Move forward slowly then backward slowly",
        ),
        (
            "move forward for two steps then stop",
            [("forward", "medium", 2), ("stop", None, 0)],
            "Move forward two steps then halt",
        ),
        (
            "move backward slowly then stop",
            [("backward", "slow", 1), ("stop", None, 0)],
            "Move backward slowly then halt",
        ),
        (
            "step forward quickly then backward quickly",
            [("forward", "fast", 1), ("backward", "fast", 1)],
            "Move forward quickly then reverse quickly",
        ),
        (
            "walk forward slowly, stop, then walk backward slowly",
            [("forward", "slow", 1), ("stop", None, 0), ("backward", "slow", 1)],
            "Sequential three-action movement",
        ),
    ]

    val_seqs = [
        (
            "advance forward slowly and then halt",
            [("forward", "slow", 1), ("stop", None, 0)],
            "Advance forward slowly then stop",
        ),
        (
            "retreat backward at medium speed then stop",
            [("backward", "medium", 1), ("stop", None, 0)],
            "Retreat backward then stop",
        ),
    ]

    test_seqs = [
        (
            "proceed forward slowly followed by backward slowly",
            [("forward", "slow", 1), ("backward", "slow", 1)],
            "Proceed forward then backward slowly",
        ),
        (
            "step forward at medium speed for three steps then stop immediately",
            [("forward", "medium", 3), ("stop", None, 0)],
            "Step forward three steps and halt",
        ),
    ]

    for split, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        reps = 25 if split == "train" else 15
        for user_cmd, acts, explanation in seqs:
            for r in range(reps):
                text = WRAPPER_TEMPLATES[r].format(cmd=user_cmd)
                action_list = []
                for idx, (tool, prof, st) in enumerate(acts, start=1):
                    params = {}
                    if prof:
                        params["speed_profile"] = prof
                    if st > 0:
                        params["steps"] = st
                    action_list.append({
                        "action_id": f"action_{idx}",
                        "tool": tool,
                        "parameters": params,
                    })
                plan = {
                    "plan_explanation": explanation,
                    "actions": action_list,
                }
                groups[split].append(format_example("ACTION_PLAN", text, plan))

    return groups


def generate_category_7_unsupported() -> Dict[str, List[Dict[str, Any]]]:
    """Category 7: Unsupported / out-of-scope tools (tool: null, reason: '...')."""
    groups = {"train": [], "val": [], "test": []}

    train_unsupported = [
        ("fly to the moon", "fly", "Tool 'fly' is not supported."),
        ("jump over the obstacle", "jump", "Tool 'jump' is not supported."),
        ("stand up on two feet", "stand", "Tool 'stand' is disabled in current stage."),
        ("sit down on the floor", "sit", "Tool 'sit' is disabled in current stage."),
        ("dance for me", "dance", "Tool 'dance' is not supported."),
        ("turn on the headlights", "headlights", "Tool 'headlights' is not supported."),
        ("wave your left arm", "wave", "Arm actuation is not supported."),
        ("lift the heavy box", "lift", "Manipulation tool 'lift' is not supported."),
    ]

    val_unsupported = [
        ("hover above the ground", "hover", "Tool 'hover' is not supported."),
        ("do a backflip", "backflip", "Acrobatic tools are not supported."),
        ("rotate your head 360 degrees", "rotate_head", "Head rotation is not supported."),
        ("pick up the ball", "pick", "Tool 'pick' is not supported."),
    ]

    test_unsupported = [
        ("soar into the sky", "soar", "Tool 'soar' is not supported."),
        ("somersault forward", "somersault", "Tool 'somersault' is not supported."),
        ("open the door", "open_door", "Manipulation tools are not supported."),
        ("climb up the ladder", "climb", "Climbing is not supported."),
    ]

    for split, items in [("train", train_unsupported), ("val", val_unsupported), ("test", test_unsupported)]:
        reps = 15 if split == "train" else 8
        for cmd, tool_hint, reason in items:
            for r in range(reps):
                text = WRAPPER_TEMPLATES[r].format(cmd=cmd)
                plan = {
                    "plan_explanation": f"Cannot fulfill request: {reason}",
                    "actions": [
                        {
                            "action_id": "action_1",
                            "tool": None,
                            "reason": reason,
                        }
                    ],
                }
                groups[split].append(format_example("UNSUPPORTED", text, plan))

    return groups


def generate_category_8_conversational() -> Dict[str, List[Dict[str, Any]]]:
    """Category 8: Conversational / Chit-Chat (actions: [])."""
    groups = {"train": [], "val": [], "test": []}

    train_conv = [
        ("hello there", "Hello! I am the IOFT robot assistant. How can I help you today?"),
        ("hi robot", "Hi! I am ready to assist with locomotion tasks."),
        ("who are you", "I am the IOFT Robot Assistant powered by Qwen 2.5 3B."),
        ("what can you do", "I can execute forward, backward, and stop locomotion actions safely."),
        ("good morning", "Good morning! Standing by for your robot commands."),
        ("thank you very much", "You are welcome! Let me know if you need anything else."),
    ]

    val_conv = [
        ("greetings assistant", "Greetings! I am ready for robot instructions."),
        ("tell me your capabilities", "I can plan forward and backward movements or stop the robot."),
        ("nice to meet you", "Nice to meet you as well! What would you like me to do?"),
    ]

    test_conv = [
        ("hey how are you doing", "I am functioning normally and ready for commands."),
        ("what is your purpose", "My purpose is to plan safe physical actions for the IOFT robot."),
        ("thanks for the help", "Glad to help! Locomotion systems are standing by."),
    ]

    conv_wrappers = [
        "{msg}",
        "{msg}!",
        "hey, {msg}",
        "{msg}, how are you?",
        "good day, {msg}",
        "hi there, {msg}",
        "hello, {msg}",
        "{msg} robot",
        "greetings, {msg}",
        "friendly hello: {msg}",
        "{msg} assistant",
        "ioft, {msg}",
        "{msg} friend",
        "salutations, {msg}",
        "ping, {msg}",
    ]

    for split, convs in [("train", train_conv), ("val", val_conv), ("test", test_conv)]:
        reps = 15 if split == "train" else 8
        for user_msg, bot_msg in convs:
            for r in range(reps):
                text = conv_wrappers[r].format(msg=user_msg)
                plan = {
                    "plan_explanation": bot_msg,
                    "actions": [],
                }
                groups[split].append(format_example("CONVERSATIONAL", text, plan))

    return groups


def main():
    print("Generating Stage 10 Fine-Tuning Dataset across 8 categories...")
    random.seed(RANDOM_SEED)

    generators = [
        generate_category_1_locomotion,
        generate_category_2_speed_profiles,
        generate_category_3_numeric_speeds,
        generate_category_4_step_counts,
        generate_category_5_clarification,
        generate_category_6_multi_action,
        generate_category_7_unsupported,
        generate_category_8_conversational,
    ]

    all_splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for gen_fn in generators:
        category_data = gen_fn()
        for split in ["train", "val", "test"]:
            all_splits[split].extend(category_data[split])

    # Shuffle each split deterministically
    for split in ["train", "val", "test"]:
        random.shuffle(all_splits[split])

    # Ensure output directories exist
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "validation"
    test_dir = OUTPUT_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write files
    split_paths = {
        "train": train_dir / "train.jsonl",
        "val": val_dir / "val.jsonl",
        "test": test_dir / "test.jsonl",
    }

    for split, path in split_paths.items():
        with open(path, "w", encoding="utf-8") as f:
            for item in all_splits[split]:
                f.write(json.dumps(item) + "\n")
        print(f"[{split.upper()}] Written {len(all_splits[split])} records to {path}")

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
