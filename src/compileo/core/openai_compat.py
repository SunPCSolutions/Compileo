"""Parameter compatibility for OpenAI Chat Completions."""

from typing import TypedDict


class SamplingOptions(TypedDict, total=False):
    temperature: float


def openai_sampling_options(model: str, temperature: float) -> SamplingOptions:
    """Keep temperature on known compatible families; otherwise use model defaults."""
    # GPT-6 variants can reject sampling overrides at their default reasoning effort.
    # An allowlist also avoids guessing capabilities for moving aliases and future IDs.
    model = model.lower()
    if model == "gpt-4" or model.startswith(
        ("gpt-3.5-turbo", "gpt-4-", "gpt-4o", "gpt-4.1", "gpt-4.5-preview")
    ):
        return {"temperature": temperature}
    return {}
