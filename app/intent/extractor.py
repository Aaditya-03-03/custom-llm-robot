"""
Intent extraction orchestrator. Combines deterministic pre-routing with Qwen 2.5 3B
single-shot extraction and fail-safe JSON/Pydantic validation.
"""

import json
import logging
import re
from typing import Optional, Dict, Any

from app.intent.schemas import (
    IntentCategory,
    RobotAction,
    MovementParameters,
    StopParameters,
    StructuredIntent,
)
from app.intent.normalizer import normalize_deterministic_intent
from app.intent.prompts import build_intent_extraction_messages
logger = logging.getLogger("custom_llm_robot.intent.extractor")


class IntentExtractor:
    """
    High-level extractor for converting natural language user instructions
    into validated, structured robot intents.
    """

    def __init__(self):
        self._llm_provider = None

    def _get_provider(self):
        if self._llm_provider is None:
            from app.llm.factory import get_llm_provider
            self._llm_provider = get_llm_provider()
        return self._llm_provider


    async def extract_intent(self, text: str) -> StructuredIntent:
        """
        Extract structured intent from user input:
        1. Fast deterministic check via normalizer (0ms LLM overhead).
        2. Qwen 2.5 3B structured extraction prompt if ambiguous.
        3. Fail-safe JSON extraction and Pydantic validation.
        """
        if not text or not text.strip():
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=text,
                is_valid=False,
                error_message="Empty input provided",
            )

        # Step 1: Fast deterministic normalizer check
        deterministic_match = normalize_deterministic_intent(text)
        if deterministic_match is not None:
            logger.info(f"Deterministic intent matched: {deterministic_match.action or deterministic_match.category}")
            return deterministic_match

        # Step 2: Dispatch to LLM for natural language interpretation
        logger.info(f"Ambiguous intent phrasing. Querying LLM for structured extraction: '{text[:60]}'")
        messages = build_intent_extraction_messages(text)

        try:
            provider = self._get_provider()
            raw_response = await provider.generate_response(messages)
            return self._parse_llm_output(raw_response, raw_input=text)
        except Exception as e:
            logger.error(f"Error during LLM intent extraction: {e}", exc_info=True)
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=text,
                is_valid=False,
                error_message=f"Extraction failed: {str(e)}",
            )

    def _parse_llm_output(self, raw_output: str, raw_input: str) -> StructuredIntent:
        """
        Robustly extract and validate JSON from LLM output string.
        Handles markdown fences, trailing commentary, and field type mismatches.
        """
        if not raw_output or not raw_output.strip():
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=raw_input,
                is_valid=False,
                error_message="LLM returned empty output",
            )

        # 1. Clean markdown fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_output.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

        # 2. Extract innermost or outermost JSON object {...}
        json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON block found in LLM response: '{raw_output[:100]}'")
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=raw_input,
                is_valid=False,
                error_message="Malformed output: no JSON object found",
            )

        json_str = json_match.group(1)

        # 3. Parse JSON
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not an object")
        except Exception as err:
            logger.warning(f"JSON decode error: {err} from text: '{json_str[:100]}'")
            return StructuredIntent(
                category=IntentCategory.UNSUPPORTED,
                action=None,
                parameters=None,
                raw_input=raw_input,
                is_valid=False,
                error_message=f"Invalid JSON format: {str(err)}",
            )

        # 4. Ingest and validate fields
        category_str = data.get("category", "unsupported")
        try:
            category = IntentCategory(category_str)
        except ValueError:
            category = IntentCategory.UNSUPPORTED

        action_str = data.get("action")
        action = None
        if action_str and category == IntentCategory.ROBOT_COMMAND:
            try:
                action = RobotAction(action_str.lower().strip())
            except ValueError:
                return StructuredIntent(
                    category=IntentCategory.UNSUPPORTED,
                    action=None,
                    parameters=None,
                    raw_input=raw_input,
                    is_valid=False,
                    error_message=f"Unknown or unsupported robot action: '{action_str}'",
                )

        # 5. Parse and validate parameters
        raw_params = data.get("parameters")
        parameters = None

        if action in (RobotAction.FORWARD, RobotAction.BACKWARD, RobotAction.LEFT, RobotAction.RIGHT):
            if isinstance(raw_params, dict):
                try:
                    speed = raw_params.get("speed")
                    steps = raw_params.get("steps")
                    if speed is not None or steps is not None:
                        parameters = MovementParameters(speed=speed, steps=steps)
                except Exception as param_err:
                    return StructuredIntent(
                        category=IntentCategory.ROBOT_COMMAND,
                        action=action,
                        parameters=None,
                        raw_input=raw_input,
                        is_valid=False,
                        error_message=f"Invalid movement parameters: {param_err}",
                    )
        elif action == RobotAction.STOP:
            if isinstance(raw_params, dict):
                parameters = StopParameters(emergency=bool(raw_params.get("emergency", False)))
            else:
                parameters = StopParameters(emergency=False)

        error_message = data.get("error_message")
        is_valid = category != IntentCategory.UNSUPPORTED

        return StructuredIntent(
            category=category,
            action=action,
            parameters=parameters,
            raw_input=raw_input,
            is_valid=is_valid,
            error_message=error_message if not is_valid else None,
        )
