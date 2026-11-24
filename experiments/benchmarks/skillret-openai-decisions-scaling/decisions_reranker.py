"""Create and validate requests for the OpenAI Decisions predicate endpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

ENDPOINT = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
CANDIDATE_COUNT = 20

# Standard Decisions rates, expressed exactly in nanodollars per token.
UNCACHED_NANO_USD = 100
CACHED_NANO_USD = 10
CACHE_WRITE_NANO_USD = 125
LONG_CONTEXT_THRESHOLD = 272_000
MODEL_CONTEXT_LIMIT = 1_050_000
NANO_USD_PER_DOLLAR = 1_000_000_000
# Cover the published long-context multiplier and regional premium on any request.
MAX_REQUEST_NANO_USD = MODEL_CONTEXT_LIMIT * CACHE_WRITE_NANO_USD * 2 * 11 // 10

_CREDENTIAL = re.compile(r"(?i)\bsk-(?:(?:proj|svcacct)-)?[\w-]{16,}")
_BEARER = re.compile(r"(?i)\bbearer[ \t]+[A-Za-z0-9._~+/=-]{16,}")
_MASKED_KEY = re.compile(r"(?i)\bsk-[\w-]+\*{2,}[\w*-]*")


class InvalidDecision(ValueError):
    """A response that cannot be used to produce a complete reranking."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_payload(
    query: str,
    candidates: list[dict[str, Any]],
    rubric: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Candidate pool has no query text.")
    if rubric["decision_type"] != "predicate":
        raise ValueError("The benchmark requires the predicate decision type.")
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError("Each candidate pool must have exactly 20 candidates.")

    ordered = sorted(candidates, key=lambda candidate: candidate["id"])
    state_candidates = {}
    questions = []
    for candidate in ordered:
        skill_id = candidate["id"]
        metadata = candidate["metadata"]
        if not isinstance(skill_id, str) or not skill_id or skill_id in state_candidates:
            raise ValueError("Candidate IDs must be nonempty and unique within a pool.")
        if not isinstance(metadata, str):
            raise ValueError("Candidate metadata must contain rendered text.")
        state_candidates[skill_id] = {"name_and_description": metadata}
        questions.append(
            {
                "type": "predicate",
                "name": skill_id,
                "instructions": rubric["instructions_template"].format(candidate_id=skill_id),
            }
        )

    return {
        "model": MODEL,
        "input": canonical_json({"query": query, "candidates": state_candidates}).decode("utf-8"),
        "questions": questions,
    }


def request_digest(encoded_payload: bytes) -> str:
    return hashlib.sha256(encoded_payload).hexdigest()


def sanitize(value: Any, api_key: str = "") -> Any:
    """Redact credentials in returned evidence without retaining request headers."""
    if isinstance(value, dict):
        return {
            sanitize(key, api_key): sanitize(item, api_key)
            for key, item in value.items()
            if key.lower() not in {"authorization", "proxy-authorization", "api_key", "api-key"}
        }
    if isinstance(value, list):
        return [sanitize(item, api_key) for item in value]
    if isinstance(value, str):
        if api_key:
            value = value.replace(api_key, "[REDACTED]")
        value = _BEARER.sub("[REDACTED]", value)
        value = _CREDENTIAL.sub("[REDACTED]", value)
        value = _MASKED_KEY.sub("[REDACTED]", value)
        # A service error body can contain lone surrogate code points.
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parse_probabilities(body: Any, candidate_ids: list[str]) -> dict[str, float]:
    if not isinstance(body, dict):
        raise InvalidDecision("non_object_response")
    model = body.get("model")
    if not isinstance(model, str) or not (
        model == MODEL or re.fullmatch(re.escape(MODEL) + r"-\d{4}-\d{2}-\d{2}", model)
    ):
        raise InvalidDecision("response_model_mismatch")

    answers = body.get("answers")
    if not isinstance(answers, list):
        raise InvalidDecision("missing_answers")
    if len(answers) != len(candidate_ids):
        raise InvalidDecision("incomplete_answers")

    expected = set(candidate_ids)
    probabilities = {}
    names = set()
    refused = False
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("type") not in {"predicate", "refusal"}:
            raise InvalidDecision("unexpected_answer_type")
        name = answer.get("name")
        if not isinstance(name, str) or name not in expected or name in names:
            raise InvalidDecision("unexpected_or_duplicate_answer_name")
        names.add(name)
        if answer["type"] == "refusal":
            refused = True
            continue
        probability = answer.get("probability")
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise InvalidDecision("invalid_probability")
        probabilities[name] = float(probability)
    if names != expected:
        raise InvalidDecision("incomplete_answers")
    if refused:
        raise InvalidDecision("predicate_refusal")
    return probabilities


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise InvalidDecision("invalid_usage")
        if (
            self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens
            or self.input_tokens > MODEL_CONTEXT_LIMIT
        ):
            raise InvalidDecision("invalid_usage")

    def cost_nano_usd(self, regional: bool) -> int:
        uncached = self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens
        cost = (
            uncached * UNCACHED_NANO_USD
            + self.cached_input_tokens * CACHED_NANO_USD
            + self.cache_write_input_tokens * CACHE_WRITE_NANO_USD
        )
        if self.input_tokens > LONG_CONTEXT_THRESHOLD:
            cost *= 2
        return (cost * 11 + 9) // 10 if regional else cost

    @classmethod
    def from_response(cls, body: Any) -> Usage:
        try:
            usage = body["usage"]
            details = usage["input_tokens_details"]
            values = cls(
                input_tokens=usage["input_tokens"],
                cached_input_tokens=details["cached_tokens"],
                cache_write_input_tokens=details["cache_write_tokens"],
                output_tokens=usage["output_tokens"],
            )
            if usage["total_tokens"] != values.input_tokens + values.output_tokens:
                raise InvalidDecision("invalid_usage")
        except (KeyError, TypeError) as error:
            raise InvalidDecision("missing_usage") from error
        return values


def reranked_candidate_ids(candidates: list[dict[str, Any]], probabilities: dict[str, float]) -> list[str]:
    return [
        candidate["id"]
        for candidate in sorted(
            candidates, key=lambda candidate: (-probabilities[candidate["id"]], candidate["rank"])
        )
    ]
