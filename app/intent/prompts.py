"""
System prompts and message builders for Qwen 2.5 3B structured intent extraction.
"""

from typing import List
from app.schemas.chat import ChatMessage

INTENT_SYSTEM_PROMPT = """\
You are an intent detection engine for a humanoid robot.
Analyze the user input and classify it into a JSON object matching this schema:

{
  "category": "conversation" | "robot_command" | "unsupported",
  "action": "forward" | "backward" | "left" | "right" | "stand" | "sit" | "stop" | "status" | "calibrate" | "manual" | null,
  "parameters": {
    "speed": int (1-100) or null,
    "steps": int (>0) or null,
    "emergency": bool (for stop) or null
  } or null,
  "error_message": string or null
}

RULES:
1. ONLY recognized robot actions: "forward", "backward", "left", "right", "stand", "sit", "stop", "status", "calibrate", "manual".
2. "left" and "right" mean directional movement left or right. Do NOT invent rotational turns.
3. If the user asks a question, chats, or asks about knowledge: category="conversation", action=null, parameters=null.
4. If the user asks for unsupported actions (fly, jump, cook, run): category="unsupported", action=null, error_message="Action unsupported".
5. OUTPUT STRICT RAW JSON ONLY. Do NOT add markdown code blocks, explanation, or conversational text.
"""

FEW_SHOT_EXAMPLES = [
    # Example 1: Movement with speed
    ChatMessage(role="user", content="Please make the robot walk forward at 50% speed."),
    ChatMessage(
        role="assistant",
        content='{"category": "robot_command", "action": "forward", "parameters": {"speed": 50, "steps": null}, "error_message": null}',
    ),
    # Example 2: Movement with steps
    ChatMessage(role="user", content="Take four steps backward."),
    ChatMessage(
        role="assistant",
        content='{"category": "robot_command", "action": "backward", "parameters": {"speed": null, "steps": 4}, "error_message": null}',
    ),
    # Example 3: Posture
    ChatMessage(role="user", content="Could you please stand up?"),
    ChatMessage(
        role="assistant",
        content='{"category": "robot_command", "action": "stand", "parameters": null, "error_message": null}',
    ),
    # Example 4: E-Stop
    ChatMessage(role="user", content="Emergency stop!"),
    ChatMessage(
        role="assistant",
        content='{"category": "robot_command", "action": "stop", "parameters": {"emergency": true}, "error_message": null}',
    ),
    # Example 5: Conversation
    ChatMessage(role="user", content="What headset is used for the robot interaction?"),
    ChatMessage(
        role="assistant",
        content='{"category": "conversation", "action": null, "parameters": null, "error_message": null}',
    ),
    # Example 6: Unsupported action
    ChatMessage(role="user", content="Make the robot fly up into the air."),
    ChatMessage(
        role="assistant",
        content='{"category": "unsupported", "action": null, "parameters": null, "error_message": "Action \'fly\' is unsupported"}',
    ),
]


def build_intent_extraction_messages(user_text: str) -> List[ChatMessage]:
    """
    Assemble the complete prompt sequence for Qwen 2.5 3B:
    [system] -> [few-shot examples] -> [current user request]
    """
    messages: List[ChatMessage] = [
        ChatMessage(role="system", content=INTENT_SYSTEM_PROMPT)
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append(ChatMessage(role="user", content=user_text))
    return messages
