"""
AI-Assisted Document Chunking Analyzer.
Handles AI analysis for optimal chunking strategy recommendations.
"""

import importlib
import json
import re
import traceback
from typing import Any, Dict, List, Optional

import streamlit as st

from src.compileo.core.openai_compat import openai_sampling_options


SUPPORTED_STRATEGIES = {"character", "token", "semantic", "delimiter", "schema"}


def _normalize_example_anchors(example_anchors: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Normalize structured example anchors."""
    normalized: List[Dict[str, str]] = []
    for example in example_anchors or []:
        if not isinstance(example, dict):
            continue

        text = str(example.get("text", "")).strip()
        example_type = str(example.get("type", "context")).strip().lower()
        if not text:
            continue
        if example_type not in {"positive", "negative", "context"}:
            example_type = "context"

        normalized.append({"type": example_type, "text": text})
    return normalized


def _build_anchor_payload(example_anchors: Optional[List[Dict[str, str]]]) -> str:
    """Serialize structured anchors into prompt-safe text."""
    anchors = _normalize_example_anchors(example_anchors)
    if not anchors:
        return ""

    sections: List[str] = []
    positive = [anchor["text"] for anchor in anchors if anchor["type"] == "positive"]
    negative = [anchor["text"] for anchor in anchors if anchor["type"] == "negative"]
    context = [anchor["text"] for anchor in anchors if anchor["type"] == "context"]

    if positive:
        sections.append("POSITIVE CHUNK START EXAMPLES:\n- " + "\n- ".join(positive[:5]))
    if negative:
        sections.append("NEGATIVE EXAMPLES (MUST NOT START CHUNKS):\n- " + "\n- ".join(negative[:5]))
    if context:
        sections.append("CONTEXT EXAMPLES:\n- " + "\n- ".join(context[:5]))

    return "\n\n=== STRUCTURED EXAMPLE ANCHORS ===\n" + "\n\n".join(sections)


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract first balanced JSON object from free-form text."""
    start_idx = -1
    for idx, char in enumerate(text):
        if char == "{":
            start_idx = idx
            break

    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:idx + 1]

    return None


def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON with tolerant extraction from model output."""
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    extracted = _extract_balanced_json(cleaned)

    candidates: List[str] = []
    if extracted:
        candidates.append(extracted)
    candidates.append(cleaned)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        for variant in [candidate, re.sub(r",\s*([}\]])", r"\1", candidate)]:
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


def _normalize_text_list(values: Any, limit: int = 5) -> List[str]:
    """Normalize arbitrary input into a bounded list of non-empty strings."""
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def _parse_confidence_value(raw_value: Any, default: float = 0.75) -> float:
    """Parse confidence value from numeric or string input."""
    try:
        confidence = float(str(raw_value).replace("%", ""))
        if confidence > 1.0:
            confidence = confidence / 100.0
        return max(0.0, min(confidence, 1.0))
    except (TypeError, ValueError):
        return default


def _anchor_signature(text: str) -> str:
    """Use the first meaningful line as the anchor signature."""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return str(text).strip()[:120]


def _extract_first_non_empty_line(text: str) -> str:
    """Extract the first non-empty line from a text block."""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _to_flexible_literal_regex(text: str) -> str:
    """Escape a literal string while allowing flexible whitespace in matching."""
    escaped = re.escape(text.strip())
    return re.sub(r"\\\s+", r"\\s+", escaped.replace(r"\ ", r"\s+"))


def _infer_common_following_markers(positive_examples: List[str], max_markers: int = 3) -> List[str]:
    """Infer common short following markers from positive examples.

    These markers are used as optional/required context constraints for candidate regex.
    """
    following_sets: List[set[str]] = []
    first_lines: set[str] = set()

    for example in positive_examples:
        lines = [line.strip() for line in str(example).splitlines() if line.strip()]
        if not lines:
            continue
        first_lines.add(lines[0])
        following = {
            line for line in lines[1:10]
            if 1 <= len(line) <= 80
        }
        if following:
            following_sets.append(following)

    if not following_sets:
        return []

    common = set.intersection(*following_sets)
    common = {
        marker for marker in common
        if marker not in first_lines
    }
    ordered = sorted(common, key=lambda marker: (len(marker), marker.lower()))
    return ordered[:max_markers]


def _count_marker_occurrences(content: str, marker: str) -> int:
    """Count how often a required marker appears in the evaluation content."""
    marker_line = _extract_first_non_empty_line(marker)
    if not content or not marker_line:
        return 0

    marker_regex = _to_flexible_literal_regex(marker_line)
    try:
        return len(re.findall(rf"(?m)^\s*{marker_regex}\s*$", content))
    except re.error:
        return 0


def _estimate_expected_chunk_count(hypothesis: Dict[str, Any], evaluation_content: str) -> Optional[int]:
    """Estimate expected chunk count from repeated required markers on real content."""
    if not evaluation_content or not bool(hypothesis.get("chunk_should_start_with_boundary", True)):
        return None

    required_markers = _normalize_text_list(hypothesis.get("required_following_markers", []), limit=4)
    marker_counts = [
        count
        for marker in required_markers
        if (count := _count_marker_occurrences(evaluation_content, marker)) > 0
    ]
    if marker_counts:
        return min(marker_counts)
    return None


def _detect_common_heading_level(start_lines: List[str]) -> Optional[str]:
    """Detect a shared markdown heading level from positive start anchors."""
    heading_levels: List[str] = []
    for line in start_lines:
        match = re.match(r"^\s*(#+)\s+\S", line)
        if not match:
            return None
        heading_levels.append(match.group(1))

    unique_levels = set(heading_levels)
    if len(unique_levels) != 1:
        return None
    return heading_levels[0]


def _line_literal(text: str) -> str:
    return r"[ \t]+".join(re.escape(word) for word in text.split())


def _build_required_marker_lookaheads(required_markers: List[str], line_budget: int = 8) -> str:
    """Require whole marker lines without crossing an unrelated Markdown heading."""
    lines = [_extract_first_non_empty_line(marker) for marker in required_markers]
    markers = [_line_literal(line) for line in lines if line]
    heading_markers = "|".join(_line_literal(line) for line in lines if re.match(r"^#{1,6}(?:\s|$)", line))
    plain_line = r"(?![ \t]*#{1,6}(?:[ \t]|\r?$))[^\r\n]*"
    allowed_line = rf"[ \t]*(?:{heading_markers})[ \t]*(?=\r?$)|{plain_line}" if heading_markers else plain_line
    gap = rf"(?:\r?\n(?:{allowed_line})){{0,{line_budget}}}"
    return "".join(rf"(?={gap}\r?\n[ \t]*{marker}[ \t]*(?=\r?$))" for marker in markers)


def _find_first_required_marker_index(lines: List[str], required_markers: List[str], start_index: int) -> Optional[int]:
    """Find the first required marker line after the boundary line."""
    normalized_markers = {
        _extract_first_non_empty_line(marker).strip().lower()
        for marker in required_markers
        if _extract_first_non_empty_line(marker).strip()
    }
    if not normalized_markers:
        return None

    for idx in range(start_index + 1, len(lines)):
        if lines[idx].strip().lower() in normalized_markers:
            return idx
    return None


def _estimate_intermediate_line_budget(
    positive_examples: List[str],
    required_markers: List[str],
    default_budget: int = 8,
    buffer_lines: int = 3,
    max_budget: int = 16,
) -> int:
    """Estimate how many non-heading lines may appear before the required marker."""
    if not positive_examples or not required_markers:
        return default_budget

    observed_gaps: List[int] = []
    for example in positive_examples:
        lines = str(example).splitlines()
        start_index = next((idx for idx, line in enumerate(lines) if line.strip()), None)
        if start_index is None:
            continue

        marker_index = _find_first_required_marker_index(lines, required_markers, start_index)
        if marker_index is None:
            continue
        observed_gaps.append(max(0, marker_index - start_index - 1))

    if not observed_gaps:
        return default_budget
    return min(max_budget, max(default_budget, max(observed_gaps) + buffer_lines))


def _infer_generalized_start_prefix(start_lines: List[str]) -> Optional[str]:
    """Infer a reusable shared textual prefix without enumerating exact titles."""
    normalized = [line.strip() for line in start_lines if str(line).strip()]
    if len(normalized) < 2:
        return None

    shared_prefix = normalized[0]
    for line in normalized[1:]:
        max_len = min(len(shared_prefix), len(line))
        idx = 0
        while idx < max_len and shared_prefix[idx].lower() == line[idx].lower():
            idx += 1
        shared_prefix = shared_prefix[:idx]
        if not shared_prefix:
            return None

    shared_prefix = shared_prefix.rstrip()
    if not shared_prefix:
        return None

    # Drop trailing variable suffixes such as chapter numbers/punctuation.
    shared_prefix = re.sub(r"[\d\W_]+$", "", shared_prefix).strip()
    if not shared_prefix:
        return None

    # Keep complete words where possible.
    if " " in shared_prefix:
        shared_prefix = shared_prefix.rsplit(" ", 1)[0].strip() or shared_prefix

    if len(shared_prefix) < 3 or not re.search(r"[A-Za-z]", shared_prefix):
        return None
    return shared_prefix


def _schema_pattern_enumerates_positive_starts(schema_obj: Dict[str, Any], positive_examples: List[str]) -> bool:
    """Detect schema patterns that enumerate exact positive example start lines."""
    start_lines: List[str] = []
    for example in _normalize_text_list(positive_examples, limit=8):
        first_line = _extract_first_non_empty_line(example)
        if first_line and first_line not in start_lines:
            start_lines.append(first_line)

    for rule in schema_obj.get("rules", []):
        if rule.get("type") != "pattern":
            continue
        pattern_value = str(rule.get("value", ""))
        for start_line in start_lines:
            if any(value in pattern_value for value in (
                re.escape(start_line), _to_flexible_literal_regex(start_line), _line_literal(start_line),
            )):
                return True
    return False


def _is_disallowed_literal_schema_candidate(candidate: Dict[str, Any], hypothesis: Dict[str, Any]) -> bool:
    """Reject schema candidates that enumerate sampled titles/examples."""
    if str(candidate.get("strategy", "")).lower() != "schema":
        return False

    schema_json = candidate.get("parameters", {}).get("json_schema")
    if not isinstance(schema_json, str):
        return False

    try:
        schema_obj = json.loads(schema_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False

    return _schema_pattern_enumerates_positive_starts(
        schema_obj,
        hypothesis.get("positive_examples", []),
    )


def _detect_boundary_hypothesis_fallback(
    sample_content: str,
    user_instructions: str,
    example_anchors: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build deterministic boundary hypothesis when model output is invalid."""
    anchors = _normalize_example_anchors(example_anchors)
    merged = f"{sample_content}\n{user_instructions}\n" + "\n".join(anchor["text"] for anchor in anchors)
    lowered = merged.lower()

    positive_examples = [anchor["text"] for anchor in anchors if anchor["type"] == "positive"]
    negative_examples = [anchor["text"] for anchor in anchors if anchor["type"] == "negative"]

    hypothesis: Dict[str, Any] = {
        "boundary_type": "heading",
        "preferred_strategy": "character",
        "chunk_should_start_with_boundary": True,
        "boundary_markers": [],
        "required_following_markers": [],
        "optional_following_markers": [],
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "simplicity_bias": "strong",
        "reasoning": "Deterministic hypothesis generated because AI output was unavailable or invalid.",
        "confidence": 0.65,
    }

    if positive_examples:
        start_lines = [_extract_first_non_empty_line(example) for example in positive_examples]
        start_lines = [line for line in start_lines if line]
        common_following = _infer_common_following_markers(positive_examples)

        heading_like = any(
            line.startswith("#")
            or line.lower().startswith("chapter")
            or line.lower().startswith("section")
            for line in start_lines
        )

        hypothesis.update({
            "boundary_type": "heading" if heading_like else "pattern",
            "preferred_strategy": "schema",
            "boundary_markers": ["markdown_h1_heading"] if heading_like else [],
            "required_following_markers": common_following[:1],
            "optional_following_markers": common_following[1:3],
            "reasoning": "Detected boundary start patterns from provided positive anchors.",
            "confidence": 0.85,
        })
        return hypothesis

    if any(keyword in lowered for keyword in ["semantic", "meaning", "topic", "concept"]):
        hypothesis.update({
            "boundary_type": "semantic",
            "preferred_strategy": "semantic",
            "reasoning": "Detected semantic or topic-oriented chunking intent.",
            "confidence": 0.72,
        })

    return hypothesis


def _normalize_boundary_hypothesis(
    raw: Dict[str, Any],
    sample_content: str,
    user_instructions: str,
    example_anchors: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Normalize model output into a constrained boundary hypothesis."""
    fallback = _detect_boundary_hypothesis_fallback(sample_content, user_instructions, example_anchors)
    anchors = _normalize_example_anchors(example_anchors)

    if not isinstance(raw, dict):
        return fallback

    if "recommended_strategy" in raw and "boundary_type" not in raw:
        preferred_strategy = str(raw.get("recommended_strategy", fallback["preferred_strategy"])).strip().lower()
        if preferred_strategy in SUPPORTED_STRATEGIES:
            fallback["preferred_strategy"] = preferred_strategy
        fallback["reasoning"] = str(raw.get("reasoning", fallback["reasoning"])).strip() or fallback["reasoning"]
        fallback["confidence"] = _parse_confidence_value(raw.get("confidence", fallback["confidence"]), fallback["confidence"])
        return fallback

    boundary_type = str(raw.get("boundary_type", fallback["boundary_type"])).strip().lower()
    if boundary_type not in {"medical_chapter_start", "heading", "delimiter", "semantic", "character", "pattern"}:
        boundary_type = fallback["boundary_type"]

    preferred_strategy = str(raw.get("preferred_strategy", fallback["preferred_strategy"])).strip().lower()
    if preferred_strategy not in SUPPORTED_STRATEGIES:
        preferred_strategy = fallback["preferred_strategy"]

    anchored_positive_examples = [anchor["text"] for anchor in anchors if anchor["type"] == "positive"]
    positive_examples = anchored_positive_examples or _normalize_text_list(
        raw.get("positive_examples") or raw.get("must_start_examples")
    )
    if not positive_examples:
        positive_examples = fallback["positive_examples"]

    anchored_negative_examples = [anchor["text"] for anchor in anchors if anchor["type"] == "negative"]
    negative_examples = anchored_negative_examples or _normalize_text_list(
        raw.get("negative_examples") or raw.get("must_not_start_examples")
    )
    if not negative_examples:
        negative_examples = fallback["negative_examples"]

    required_following_markers = _normalize_text_list(raw.get("required_following_markers"))
    if not required_following_markers:
        required_following_markers = fallback["required_following_markers"]

    optional_following_markers = _normalize_text_list(raw.get("optional_following_markers"))
    if not optional_following_markers:
        optional_following_markers = fallback["optional_following_markers"]

    boundary_markers = _normalize_text_list(raw.get("boundary_markers"))
    if not boundary_markers:
        boundary_markers = fallback["boundary_markers"]

    if boundary_type == "medical_chapter_start" and "# BASICS" not in required_following_markers:
        required_following_markers.append("# BASICS")

    chunk_should_start_with_boundary = raw.get("chunk_should_start_with_boundary", True)
    if not isinstance(chunk_should_start_with_boundary, bool):
        chunk_should_start_with_boundary = True

    simplicity_bias = str(raw.get("simplicity_bias", fallback["simplicity_bias"])).strip().lower()
    if simplicity_bias not in {"strong", "medium", "low"}:
        simplicity_bias = "strong"

    return {
        "boundary_type": boundary_type,
        "preferred_strategy": preferred_strategy,
        "chunk_should_start_with_boundary": chunk_should_start_with_boundary,
        "boundary_markers": boundary_markers,
        "required_following_markers": required_following_markers,
        "optional_following_markers": optional_following_markers,
        "alternative_following_markers": _normalize_text_list(raw.get("alternative_following_markers"), limit=4),
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "simplicity_bias": simplicity_bias,
        "reasoning": str(raw.get("reasoning", fallback["reasoning"])).strip() or fallback["reasoning"],
        "confidence": _parse_confidence_value(raw.get("confidence", fallback["confidence"]), fallback["confidence"]),
    }


def _build_boundary_hypothesis_prompt(
    sample_content: str,
    user_instructions: str,
    example_anchors: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the structured boundary-hypothesis prompt for AI analysis."""
    anchor_payload = _build_anchor_payload(example_anchors)
    return f"""You are an expert document chunking analyst.

Your job is NOT to output final regex or final schema.
Describe major chunk starts, not every subheading. Code compiles your hypothesis and checks it on the full evaluation scope.

USER REQUIREMENTS:
{user_instructions}

DOCUMENT SAMPLE:
{sample_content}
{anchor_payload}

AVAILABLE STRATEGIES:
- character
- token
- semantic
- delimiter
- schema

RETURN EXACTLY ONE JSON OBJECT WITH THIS STRUCTURE:
{{
  "boundary_type": "heading|delimiter|semantic|character|pattern",
  "preferred_strategy": "schema|semantic|delimiter|character|token",
  "chunk_should_start_with_boundary": true,
  "boundary_markers": ["markdown_h1_heading"],
  "required_following_markers": ["# BASICS"],
  "alternative_following_markers": [],
  "positive_examples": ["exact examples that SHOULD start chunks"],
  "negative_examples": ["exact examples that SHOULD NOT start chunks"],
  "simplicity_bias": "strong|medium|low",
  "reasoning": "short explanation",
  "confidence": 0.0
}}

RULES:
- Prefer the simplest useful boundary hypothesis.
- Infer a shared reusable structure from examples; examples are illustrative, not exhaustive.
- Do NOT output regex.
- Do NOT output json_schema.
- Keep the hypothesis domain-agnostic; do not assume medical/legal/technical unless explicitly shown in sample or anchors.
- When positive examples exist, treat them as hard boundary anchors.
- Do NOT restate sampled titles, entity names, or section names as the rule unless they are repeated structural markers.
- Prefer reusable descriptors such as heading level, nearby repeated markers, delimiter style, or semantic cues.
- Required markers are exact recurring lines that must all be present. Supply alternative markers only for genuinely distinct layouts where any one alternative is sufficient. Do not list whitespace variations as separate layouts.
- Use the structured anchors when present.
- If uncertain, choose simpler hypotheses rather than complex ones.

Return JSON only.
"""


def _trim_scored_chunks(chunks: List[str]) -> List[str]:
    """Trim display whitespace without dropping substantive chunks."""
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _anchor_occurrences(content: str, examples: List[str]) -> List[set[int]]:
    """Locate exact lines plus any supplied context, allowing blank-line variations."""
    occurrences = []
    for example in examples:
        lines = [_line_literal(line) for line in example.splitlines() if line.strip()]
        expression = r"[ \t]*\r?\n(?:[ \t]*\r?\n)*[ \t]*".join(lines)
        occurrences.append({
            match.start(1)
            for match in re.finditer(rf"^[ \t]*(?=({expression})[ \t]*\r?$)", content, re.MULTILINE)
        } if lines else set())
    return occurrences


def _score_chunk_lengths(chunks: List[str]) -> float:
    """Score chunk size quality with simplicity and utility bias."""
    cleaned = _trim_scored_chunks(chunks)
    if not cleaned:
        return 0.0

    lengths = [len(chunk) for chunk in cleaned]
    median_length = sorted(lengths)[len(lengths) // 2]
    tiny_chunks = sum(1 for length in lengths if length < 250)
    oversized_chunks = sum(1 for length in lengths if length > 12000)

    score = 1.0
    if median_length < 400:
        score -= 0.35
    elif median_length < 800:
        score -= 0.15
    if len(cleaned) > 20:
        score -= 0.15
    score -= min(0.4, tiny_chunks * 0.12)
    score -= min(0.25, oversized_chunks * 0.08)
    return max(0.0, min(score, 1.0))


def _extract_chunk_start_lines(chunks: List[str], limit: int = 5) -> List[str]:
    """Extract representative chunk start lines for preview/debugging."""
    starts: List[str] = []
    for chunk in _trim_scored_chunks(chunks):
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped:
                starts.append(stripped[:160])
                break
        if len(starts) >= limit:
            break
    return starts


def _count_anchor_hits(start_lines: List[str], examples: List[str]) -> int:
    """Count how many example anchors appear in candidate start lines."""
    signatures = [_anchor_signature(example) for example in examples if _anchor_signature(example)]
    hits = 0
    for signature in signatures:
        lowered = signature.lower()
        if any(lowered in line.lower() or line.lower() in lowered for line in start_lines):
            hits += 1
    return hits


def _chunk_with_character(content: str, chunk_size: int, overlap: int) -> List[str]:
    """Production-aligned character chunking used for recommendation dry-runs."""
    if not content.strip() or chunk_size <= 0 or overlap >= chunk_size:
        return [content.strip()] if content.strip() else []

    from src.compileo.features.chunk.engine import chunk_document
    from src.compileo.features.chunk.schema import CharacterStrategy

    strategy = CharacterStrategy(strategy="character", chunk_size=chunk_size, overlap=overlap)
    return chunk_document(content, strategy)


def _chunk_with_delimiter(content: str, delimiter: str) -> List[str]:
    """Production-aligned delimiter chunking used for recommendation dry-runs."""
    if not delimiter:
        return [content.strip()] if content.strip() else []

    from src.compileo.features.chunk.engine import chunk_document
    from src.compileo.features.chunk.schema import DelimiterStrategy

    strategy = DelimiterStrategy(strategy="delimiter", delimiter=delimiter)
    return chunk_document(content, strategy)


def _score_schema_candidate(
    schema_obj: Dict[str, Any],
    sample_content: str,
    hypothesis: Dict[str, Any],
    candidate_name: str,
    candidate_family: str = "schema",
    generalization_score: float = 0.5,
    literal_specificity: float = 0.5,
) -> Dict[str, Any]:
    """Dry-run a schema candidate against sample content and compute quality score."""
    from ...chunk.engine import schema_chunk_spans
    from ...chunk.schema import SchemaStrategy

    schema_json = json.dumps(schema_obj)
    try:
        spans = schema_chunk_spans(sample_content, SchemaStrategy(json_schema=schema_json))
    except ValueError as exc:
        return {
            "candidate_name": candidate_name,
            "strategy": "schema",
            "parameters": {},
            "score": 0.0,
            "chunk_count": 0,
            "preview_start_lines": [],
            "positive_anchor_hits": 0,
            "negative_anchor_hits": 0,
            "complexity_penalty": 1.0,
            "candidate_family": candidate_family,
            "generalization_score": round(generalization_score, 4),
            "literal_specificity": round(literal_specificity, 4),
            "details": str(exc),
        }

    cleaned = [sample_content[span.start:span.end] for span in spans]
    boundary_starts = {span.start for span in spans if span.matched_rule}
    positive = _anchor_occurrences(sample_content, hypothesis.get("positive_examples", []))
    negative = _anchor_occurrences(sample_content, hypothesis.get("negative_examples", []))
    positive_hits = sum(len(offsets) == 1 and bool(offsets & boundary_starts) for offsets in positive)
    actual_starts = {span.start for span in spans}
    negative_hits = sum(bool(offsets & actual_starts) for offsets in negative)

    score = _score_chunk_lengths(cleaned)
    if hypothesis.get("positive_examples"):
        score += 0.35 * (positive_hits / max(1, len(hypothesis["positive_examples"])))
    if hypothesis.get("negative_examples"):
        score -= 0.35 * (negative_hits / max(1, len(hypothesis["negative_examples"])))

    complexity_penalty = max(0.0, (len(schema_obj.get("rules", [])) - 1) * 0.08)
    score -= complexity_penalty

    score = max(0.0, min(score, 1.0))
    return {
        "candidate_name": candidate_name,
        "strategy": "schema",
        "parameters": {"json_schema": schema_json},
        "score": round(score, 4),
        "chunk_count": len(cleaned),
        "preview_start_lines": _extract_chunk_start_lines(cleaned),
        "boundary_offsets": sorted(boundary_starts),
        "anchor_resolution_issues": sum(len(offsets) != 1 for offsets in positive) + sum(not offsets for offsets in negative),
        "positive_anchor_hits": positive_hits,
        "negative_anchor_hits": negative_hits,
        "complexity_penalty": round(complexity_penalty, 4),
        "candidate_family": candidate_family,
        "generalization_score": round(generalization_score, 4),
        "literal_specificity": round(literal_specificity, 4),
        "details": f"{len(cleaned)} chunks in dry-run; {positive_hits} positive hits; {negative_hits} negative hits.",
    }


def _score_character_candidate(
    sample_content: str,
    chunk_size: int,
    overlap: int,
    candidate_name: str,
    hypothesis: Dict[str, Any],
    candidate_family: str = "character_fallback",
    generalization_score: float = 0.0,
    literal_specificity: float = 0.0,
) -> Dict[str, Any]:
    """Dry-run a character strategy candidate."""
    chunks = _chunk_with_character(sample_content, chunk_size, overlap)
    cleaned = _trim_scored_chunks(chunks)
    preview_lines = _extract_chunk_start_lines(cleaned, limit=max(5, len(cleaned)))
    positive_hits = _count_anchor_hits(preview_lines, hypothesis.get("positive_examples", []))
    negative_hits = _count_anchor_hits(preview_lines, hypothesis.get("negative_examples", []))

    return {
        "candidate_name": candidate_name,
        "strategy": "character",
        "parameters": {"chunk_size": chunk_size, "overlap": overlap},
        "score": round(_score_chunk_lengths(cleaned), 4),
        "chunk_count": len(cleaned),
        "preview_start_lines": preview_lines[:5],
        "positive_anchor_hits": positive_hits,
        "negative_anchor_hits": negative_hits,
        "complexity_penalty": 0.0,
        "candidate_family": candidate_family,
        "generalization_score": round(generalization_score, 4),
        "literal_specificity": round(literal_specificity, 4),
        "details": f"{len(cleaned)} chunks in dry-run; {positive_hits} positive hits; {negative_hits} negative hits.",
    }


def _score_delimiter_candidate(
    sample_content: str,
    delimiter: str,
    candidate_name: str,
    hypothesis: Dict[str, Any],
    candidate_family: str = "delimiter",
    generalization_score: float = 0.45,
    literal_specificity: float = 0.1,
) -> Dict[str, Any]:
    """Dry-run a delimiter strategy candidate."""
    chunks = _chunk_with_delimiter(sample_content, delimiter)
    cleaned = _trim_scored_chunks(chunks)
    preview_lines = _extract_chunk_start_lines(cleaned, limit=max(5, len(cleaned)))
    positive_hits = _count_anchor_hits(preview_lines, hypothesis.get("positive_examples", []))
    negative_hits = _count_anchor_hits(preview_lines, hypothesis.get("negative_examples", []))

    return {
        "candidate_name": candidate_name,
        "strategy": "delimiter",
        "parameters": {"delimiters": [delimiter]},
        "score": round(_score_chunk_lengths(cleaned), 4),
        "chunk_count": len(cleaned),
        "preview_start_lines": preview_lines[:5],
        "positive_anchor_hits": positive_hits,
        "negative_anchor_hits": negative_hits,
        "complexity_penalty": 0.0,
        "candidate_family": candidate_family,
        "generalization_score": round(generalization_score, 4),
        "literal_specificity": round(literal_specificity, 4),
        "details": f"{len(cleaned)} chunks in dry-run using delimiter {delimiter!r}; {positive_hits} positive hits; {negative_hits} negative hits.",
    }


def _build_semantic_candidate(hypothesis: Dict[str, Any], user_instructions: str, candidate_name: str) -> Dict[str, Any]:
    """Create a semantic candidate without additional model calls."""
    semantic_prompt = user_instructions.strip()
    positive_examples = hypothesis.get("positive_examples", [])
    negative_examples = hypothesis.get("negative_examples", [])

    if positive_examples:
        semantic_prompt += "\n\nPositive chunk-start anchors:\n- " + "\n- ".join(positive_examples[:5])
    if negative_examples:
        semantic_prompt += "\n\nAvoid starting chunks at:\n- " + "\n- ".join(negative_examples[:5])

    score = min(0.65, 0.3 + 0.2 * hypothesis.get("confidence", 0.7))
    return {
        "candidate_name": candidate_name,
        "strategy": "semantic",
        "parameters": {"semantic_prompt": semantic_prompt},
        "score": round(score, 4),
        "chunk_count": 0,
        "preview_start_lines": [],
        "positive_anchor_hits": 0,
        "negative_anchor_hits": 0,
        "complexity_penalty": 0.04,
        "candidate_family": "semantic",
        "generalization_score": 0.55,
        "literal_specificity": 0.0,
        "details": "Semantic candidate selected from boundary hypothesis; no additional model dry-run performed.",
    }


def _build_anchor_based_schema_candidates(
    hypothesis: Dict[str, Any],
    sample_content: str,
) -> List[Dict[str, Any]]:
    """Build schema candidates directly from positive/negative anchor information.

    This keeps the approach regex-based but domain-agnostic.
    """
    candidates: List[Dict[str, Any]] = []
    positive_examples = _normalize_text_list(hypothesis.get("positive_examples", []), limit=6)
    start_lines = [_extract_first_non_empty_line(example) for example in positive_examples]
    start_lines = [line for line in start_lines if line]
    unique_start_lines = []
    for line in start_lines:
        if line not in unique_start_lines:
            unique_start_lines.append(line)

    if not unique_start_lines:
        return candidates

    common_heading_level = _detect_common_heading_level(unique_start_lines)
    marker_options = _marker_options(hypothesis)
    if common_heading_level and any(marker_options):
        heading = _heading_pattern(common_heading_level)
        candidates.append(_score_schema_candidate(
            _structural_schema(heading, marker_options, positive_examples),
            sample_content,
            hypothesis,
            "Structural heading level + required marker",
            candidate_family="structural_heading_marker",
            generalization_score=0.95,
            literal_specificity=0.1,
        ))

    if common_heading_level and not any(marker_options):
        candidates.append(_score_schema_candidate(
            _structural_schema(_heading_pattern(common_heading_level), marker_options, positive_examples),
            sample_content,
            hypothesis,
            "Structural heading level starts",
            candidate_family="structural_heading",
            generalization_score=0.75,
            literal_specificity=0.05,
        ))

    # Generalized prefix candidate when heading-level inference is unavailable.
    generalized_prefix = _infer_generalized_start_prefix(unique_start_lines)
    if generalized_prefix and not common_heading_level:
        prefix_regex = _line_literal(generalized_prefix)
        generalized_pattern = rf"^[ \t]*{prefix_regex}\b[^\r\n]*(?=\r?$)"
        candidates.append(_score_schema_candidate(
            _structural_schema(generalized_pattern, marker_options, positive_examples),
            sample_content,
            hypothesis,
            "Anchor-generalized prefix starts",
            candidate_family="structural_prefix",
            generalization_score=0.65,
            literal_specificity=0.2,
        ))

    return candidates


def _heading_pattern(level: str) -> str:
    return rf"^[ \t]*{re.escape(level)}[ \t]+[^ \t\r\n][^\r\n]*(?=\r?$)"


def _marker_options(hypothesis: Dict[str, Any]) -> List[List[str]]:
    required = _normalize_text_list(hypothesis.get("required_following_markers"), limit=4)
    alternatives = _normalize_text_list(hypothesis.get("alternative_following_markers"), limit=4)
    return [list(dict.fromkeys([*required, marker])) for marker in alternatives] or [required]


def _structural_schema(prefix: str, options: List[List[str]], examples: List[str]) -> Dict[str, Any]:
    patterns = [
        prefix + _build_required_marker_lookaheads(markers, _estimate_intermediate_line_budget(examples, markers))
        for markers in options
    ]
    return {
        "rules": [{"type": "pattern", "value": pattern} for pattern in dict.fromkeys(patterns)],
        "combine": "any",
        "include_pattern": True,
    }


def _compile_candidates_from_hypothesis(
    hypothesis: Dict[str, Any],
    sample_content: str,
    user_instructions: str,
) -> List[Dict[str, Any]]:
    """Compile multiple simple candidates from a boundary hypothesis."""
    candidates: List[Dict[str, Any]] = []
    boundary_type = hypothesis.get("boundary_type")
    preferred_strategy = hypothesis.get("preferred_strategy", "schema")
    boundary_markers = [marker.lower() for marker in hypothesis.get("boundary_markers", [])]
    marker_options = _marker_options(hypothesis)
    examples = hypothesis.get("positive_examples", [])

    anchor_based_candidates = _build_anchor_based_schema_candidates(hypothesis, sample_content)
    candidates.extend(anchor_based_candidates)

    if boundary_type == "delimiter" or preferred_strategy == "delimiter":
        delimiter_value = "# " if "markdown_h1_heading" in boundary_markers else "#"
        candidates.append(_score_delimiter_candidate(sample_content, delimiter_value, "Delimiter-based header split", hypothesis))

    if boundary_type == "heading" or preferred_strategy == "schema":
        level = next((f"{'#' * n}" for n in range(1, 7) if f"markdown_h{n}_heading" in boundary_markers), "#")
        candidates.append(_score_schema_candidate(
            _structural_schema(_heading_pattern(level), marker_options, examples),
            sample_content,
            hypothesis,
            "General markdown heading starts",
            candidate_family="generic_heading",
            generalization_score=0.7 if any(marker_options) else 0.6,
            literal_specificity=0.05,
        ))

    if preferred_strategy == "semantic" or boundary_type == "semantic":
        candidates.append(_build_semantic_candidate(hypothesis, user_instructions, "Semantic boundary hypothesis"))

    # Hard guard: never allow literal sampled-title schema candidates into scoring/UI.
    candidates = [
        candidate
        for candidate in candidates
        if not _is_disallowed_literal_schema_candidate(candidate, hypothesis)
    ]

    candidates.append(_score_character_candidate(sample_content, 1200, 120, "Character fallback baseline", hypothesis))
    return candidates


def _select_best_candidate(
    hypothesis: Dict[str, Any], sample_content: str, user_instructions: str, *, multi_file: bool = False,
) -> Dict[str, Any]:
    """Compile, score, and select the best recommendation candidate."""
    candidates = _compile_candidates_from_hypothesis(hypothesis, sample_content, user_instructions)
    preferred_strategy = str(hypothesis.get("preferred_strategy", "")).lower()
    boundary_type = str(hypothesis.get("boundary_type", "")).lower()
    positive_examples = hypothesis.get("positive_examples", [])
    negative_examples = hypothesis.get("negative_examples", [])
    chunk_should_start_with_boundary = bool(hypothesis.get("chunk_should_start_with_boundary", True))
    marker_count_hint = _estimate_expected_chunk_count(hypothesis, sample_content)

    structured_request = (
        preferred_strategy in {"schema", "delimiter"}
        or boundary_type in {"heading", "delimiter", "medical_chapter_start", "pattern"}
    )

    qualified_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        strategy = str(candidate.get("strategy", "")).lower()
        positive_hits = int(candidate.get("positive_anchor_hits", 0))
        negative_hits = int(candidate.get("negative_anchor_hits", 0))

        positive_coverage = (
            positive_hits / max(1, len(positive_examples))
            if positive_examples
            else 1.0
        )

        disqualification_reasons: List[str] = []
        if positive_examples and positive_coverage < 1:
            disqualification_reasons.append(
                f"verified {positive_hits}/{len(positive_examples)} positive anchors"
            )
        if candidate.get("anchor_resolution_issues"):
            disqualification_reasons.append("Anchor text is absent or ambiguous; add surrounding lines or select the right files")
        if negative_examples and negative_hits > 0:
            disqualification_reasons.append(
                f"matched negative anchors ({negative_hits})"
            )
        if structured_request and strategy in {"character", "token"}:
            disqualification_reasons.append(
                "fallback strategy blocked for structured boundary request"
            )
        if strategy == "schema" and (not candidate.get("boundary_offsets") or not chunk_should_start_with_boundary):
            disqualification_reasons.append("No rule-backed starts matching the requested placement")
        if strategy == "semantic":
            disqualification_reasons.append("Semantic boundaries were not dry-run; use manual semantic configuration")
        if strategy != "schema" and (positive_examples or negative_examples):
            disqualification_reasons.append("This candidate does not verify anchor occurrences")
        if multi_file and strategy != "schema":
            disqualification_reasons.append("Only schema candidates are verified for whole-document analysis")
        if not candidate.get("chunk_count"):
            disqualification_reasons.append(candidate.get("details", "No chunks in dry-run"))

        candidate["positive_coverage"] = round(positive_coverage, 4)
        candidate["qualified"] = len(disqualification_reasons) == 0
        candidate["disqualification_reasons"] = disqualification_reasons

        if candidate["qualified"]:
            qualified_candidates.append(candidate)

    if not qualified_candidates:
        return {
            "recommended_strategy": None,
            "parameters": {},
            "reasoning": "No verified rule satisfies the boundary constraints on the evaluated files.",
            "boundary_hypothesis": hypothesis,
            "candidate_evaluations": candidates,
        }

    def _count_match_score(candidate: Dict[str, Any]) -> float:
        if marker_count_hint is None or hypothesis.get("alternative_following_markers"):
            return 0.0
        distance = abs(len(candidate.get("boundary_offsets", [])) - marker_count_hint)
        return max(0.0, 1.0 - distance / marker_count_hint)

    def _rank(candidate: Dict[str, Any]) -> tuple[float, float, float, int, float, float, float]:
        preferred_bonus = 1 if str(candidate.get("strategy", "")).lower() == preferred_strategy else 0
        return (
            float(candidate.get("positive_coverage", 1.0)),
            float(candidate.get("generalization_score", 0.0)),
            _count_match_score(candidate),
            preferred_bonus,
            -int(candidate.get("negative_anchor_hits", 0)),
            -float(candidate.get("literal_specificity", 0.0)),
            float(candidate.get("score", 0.0)) - float(candidate.get("complexity_penalty", 0.0)),
        )

    best = max(qualified_candidates, key=_rank)
    return {
        "recommended_strategy": best["strategy"],
        "parameters": best["parameters"],
        "reasoning": hypothesis.get("reasoning", "Boundary hypothesis generated.") + f" Selected candidate: {best['candidate_name']} based on dry-run scoring.",
        "confidence": min(0.98, max(0.4, best.get("score", 0.0))),
        "boundary_hypothesis": hypothesis,
        "candidate_evaluations": candidates,
        "selected_candidate": best["candidate_name"],
        "preview_start_lines": best.get("preview_start_lines", []),
        "boundary_offsets": best.get("boundary_offsets", []),
        "estimated_chunk_count": best.get("chunk_count", 0),
    }


def _build_deterministic_fallback(sample_content: str, user_instructions: str) -> Dict[str, Any]:
    """Backward-compatible deterministic fallback wrapper."""
    hypothesis = _detect_boundary_hypothesis_fallback(sample_content, user_instructions)
    return _select_best_candidate(hypothesis, sample_content, user_instructions)


def _normalize_recommendation(raw: Dict[str, Any], sample_content: str, user_instructions: str) -> Optional[Dict[str, Any]]:
    """Backward-compatible normalization wrapper."""
    if not isinstance(raw, dict):
        return None
    hypothesis = _normalize_boundary_hypothesis(raw, sample_content, user_instructions)
    return _select_best_candidate(hypothesis, sample_content, user_instructions)


def analyze_document_for_chunking(
    sample_content: str,
    user_instructions: str,
    selected_chunker: str,
    example_anchors: Optional[List[Dict[str, str]]] = None,
    evaluation_content: Optional[str] = None,
    evaluation_parts: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze document sample and provide chunking recommendations using AI."""
    try:
        from src.compileo.core.settings import BackendSettings
        from ..state.session_state import session_state

        user_settings = BackendSettings(owner_id=session_state.user_id)
        model_name = None
        api_key = None

        if selected_chunker == "gemini":
            api_key = user_settings.get_gemini_api_key()
            model_name = user_settings.get_chunking_gemini_model()
        elif selected_chunker == "grok":
            api_key = user_settings.get_grok_api_key()
            model_name = user_settings.get_chunking_grok_model()
        elif selected_chunker == "openai":
            api_key = user_settings.get_openai_api_key()
            model_name = user_settings.get_chunking_openai_model()
        elif selected_chunker == "ollama":
            model_name = user_settings.get_chunking_ollama_model()

        if not model_name:
            st.error("❌ No chunking model configured. Please check your settings.")
            return None
        if selected_chunker in {"gemini", "grok", "openai"} and not api_key:
            st.error(f"❌ Missing API key for {selected_chunker}. Please configure it in settings.")
            return None

        analysis_prompt = _build_boundary_hypothesis_prompt(sample_content, user_instructions, example_anchors)
        from ...chunk.engine import join_schema_parts

        parts = evaluation_parts if evaluation_parts is not None else [
            evaluation_content if evaluation_content is not None else sample_content
        ]
        scoring_content = join_schema_parts(parts)
        multi_file = len(parts) > 1

        response = None
        if selected_chunker == "gemini":
            response = call_gemini_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "grok":
            response = call_grok_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "openai":
            response = call_openai_api(analysis_prompt, api_key, model_name)
        elif selected_chunker == "ollama":
            response = call_ollama_api(analysis_prompt, model_name)

        if not response:
            hypothesis = _detect_boundary_hypothesis_fallback(sample_content, user_instructions, example_anchors)
            st.warning("AI response unavailable. Evaluating a deterministic boundary hypothesis.")
            return _select_best_candidate(hypothesis, scoring_content, user_instructions, multi_file=multi_file)

        parsed = _parse_json_loose(response)
        if not parsed:
            hypothesis = _detect_boundary_hypothesis_fallback(sample_content, user_instructions, example_anchors)
            st.warning("AI response was invalid. Evaluating a deterministic boundary hypothesis.")
            return _select_best_candidate(hypothesis, scoring_content, user_instructions, multi_file=multi_file)

        hypothesis = _normalize_boundary_hypothesis(parsed, sample_content, user_instructions, example_anchors)
        return _select_best_candidate(hypothesis, scoring_content, user_instructions, multi_file=multi_file)

    except Exception as e:
        st.error(f"❌ Error during AI analysis: {str(e)}")
        st.error(traceback.format_exc())
        return None


def _display_recommendation_parameters(strategy: Optional[str], parameters: Dict[str, Any]) -> bool:
    """Render parameters and return whether the schema, if any, can be applied."""
    if strategy == "schema":
        from ...chunk.schema import parse_chunking_schema

        schema_json = parameters.get("json_schema")
        try:
            parse_chunking_schema(schema_json)
        except ValueError as exc:
            st.error(f"Cannot use the recommended schema: {exc}")
            return False
        # Markdown can consume backslashes and other characters in regex JSON.
        st.code(schema_json, language="json")

    params_display = "\n".join(
        f"- **{key}**: {value}" for key, value in parameters.items() if key != "json_schema"
    )
    if params_display or strategy != "schema":
        st.markdown(params_display or "- *(none)*")
    return True


def display_ai_recommendations(recommendations: Dict[str, Any]) -> None:
    """Display AI recommendations and provide accept/adjust options."""
    st.markdown("### 🎯 AI Recommendations")

    strategy = recommendations.get("recommended_strategy", "")
    parameters = recommendations.get("parameters", {})
    reasoning = recommendations.get("reasoning", "")
    confidence = recommendations.get("confidence", 0)
    selected_candidate = recommendations.get("selected_candidate")
    estimated_chunk_count = recommendations.get("estimated_chunk_count")
    preview_start_lines = recommendations.get("preview_start_lines", [])
    candidate_evaluations = recommendations.get("candidate_evaluations", [])
    boundary_hypothesis = recommendations.get("boundary_hypothesis", {})

    if not strategy:
        st.warning("No verified rule. Review the candidate diagnostics or use manual configuration.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Recommended Strategy:** {strategy or 'None'}")
        try:
            confidence_val = float(confidence) if isinstance(confidence, str) else confidence
            st.markdown(f"**Confidence:** {confidence_val:.1%}")
        except (ValueError, TypeError):
            st.markdown(f"**Confidence:** {confidence}")
    with col2:
        try:
            confidence_val = float(confidence) if isinstance(confidence, str) else confidence
            if confidence_val >= 0.8:
                st.markdown("🟢 **High Confidence**")
            elif confidence_val >= 0.6:
                st.markdown("🟡 **Medium Confidence**")
            else:
                st.markdown("🔴 **Low Confidence**")
        except (ValueError, TypeError):
            st.markdown("⚪ **Confidence Unknown**")

    with st.expander("📋 AI Reasoning"):
        st.markdown(reasoning)

    st.markdown("**📊 Recommended Parameters:**")
    recommendation_valid = _display_recommendation_parameters(strategy, parameters)

    if selected_candidate:
        st.markdown(f"**Selected Candidate:** {selected_candidate}")
    if estimated_chunk_count is not None:
        st.markdown(f"**Estimated Chunk Count (evaluation dry-run):** {estimated_chunk_count}")

    if preview_start_lines:
        with st.expander("🔎 Dry-Run Boundary Preview"):
            st.markdown("**First detected chunk starts on evaluation content:**")
            for line in preview_start_lines:
                st.markdown(f"- `{line}`")

    if candidate_evaluations:
        with st.expander("🧪 Candidate Scoring"):
            score_rows = []
            for candidate in candidate_evaluations:
                score_rows.append({
                    "candidate": candidate.get("candidate_name", "unknown"),
                    "strategy": candidate.get("strategy", "unknown"),
                    "score": candidate.get("score", 0.0),
                    "chunks": candidate.get("chunk_count", 0),
                    "family": candidate.get("candidate_family", "unknown"),
                    "positive_hits": candidate.get("positive_anchor_hits", 0),
                    "positive_coverage": candidate.get("positive_coverage", 0.0),
                    "negative_hits": candidate.get("negative_anchor_hits", 0),
                    "generalization": candidate.get("generalization_score", 0.0),
                    "literal_specificity": candidate.get("literal_specificity", 0.0),
                    "complexity_penalty": candidate.get("complexity_penalty", 0.0),
                    "qualified": candidate.get("qualified", True),
                })
            st.table(score_rows)

            disqualified = [
                candidate
                for candidate in candidate_evaluations
                if not candidate.get("qualified", True)
            ]
            if disqualified:
                st.markdown("**Disqualified candidates:**")
                for candidate in disqualified:
                    reasons = candidate.get("disqualification_reasons", [])
                    reason_text = "; ".join(reasons) if reasons else "unknown reason"
                    st.markdown(
                        f"- `{candidate.get('candidate_name', 'unknown')}`: {reason_text}"
                    )

    applicable = [candidate for candidate in candidate_evaluations if candidate.get("qualified")]
    if applicable:
        with st.expander("🎛️ Choose Candidate Manually"):
            candidate_labels = []
            for idx, candidate in enumerate(applicable):
                candidate_labels.append(
                    f"{idx}: {candidate.get('candidate_name', 'unknown')} | "
                    f"{candidate.get('strategy', 'unknown')} | "
                    f"score={candidate.get('score', 0.0):.3f} | "
                    f"chunks={candidate.get('chunk_count', 0)}"
                )

            selected_label = st.selectbox(
                "Apply a different candidate instead of the auto-selected winner:",
                options=candidate_labels,
                key="ai_candidate_selector",
            )
            selected_index = int(selected_label.split(":", 1)[0])
            selected_candidate_data = applicable[selected_index]

            candidate_strategy = selected_candidate_data.get("strategy", strategy)
            selected_params = selected_candidate_data.get("parameters", {})
            st.markdown(f"**Candidate Parameters:** `{candidate_strategy}`")
            candidate_valid = _display_recommendation_parameters(candidate_strategy, selected_params)

            if st.button(
                "✅ Apply Selected Candidate", key="apply_selected_candidate", disabled=not candidate_valid
            ) and apply_ai_recommendations(candidate_strategy, selected_params):
                st.success("✅ Selected candidate applied to configuration!")
                st.rerun()

    if boundary_hypothesis:
        with st.expander("🧭 Boundary Hypothesis"):
            st.json(boundary_hypothesis)

    if not strategy:
        return

    col_accept, col_adjust, col_refine = st.columns([1, 1, 1])
    with col_accept:
        if st.button(
            "✅ Accept & Apply", type="primary", disabled=not recommendation_valid
        ) and apply_ai_recommendations(strategy, parameters):
            st.success("✅ Recommendations applied to configuration!")
            st.rerun()

    with col_adjust:
        if strategy != "schema" and st.button("🔧 Adjust Parameters"):
            st.session_state.show_parameter_adjustment = True

    with col_refine:
        if st.button("🔄 Refine Analysis"):
            st.session_state.refine_analysis = True

    if strategy != "schema" and st.session_state.get("show_parameter_adjustment", False):
        st.markdown("### 🔧 Adjust Parameters")
        adjusted_params: Dict[str, Any] = {}
        for param_name, param_value in parameters.items():
            if isinstance(param_value, int):
                adjusted_params[param_name] = st.slider(
                    f"{param_name}",
                    min_value=max(1, param_value // 2),
                    max_value=max(2, param_value * 2),
                    value=param_value,
                )
            elif isinstance(param_value, float):
                adjusted_params[param_name] = st.slider(
                    f"{param_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=param_value,
                    step=0.1,
                )
            else:
                adjusted_params[param_name] = st.text_input(f"{param_name}", value=str(param_value))

        if st.button("✅ Apply Adjusted Parameters") and apply_ai_recommendations(strategy, adjusted_params):
            st.success("✅ Adjusted parameters applied!")
            st.session_state.show_parameter_adjustment = False
            st.rerun()


def apply_ai_recommendations(strategy: str, parameters: Dict[str, Any]) -> bool:
    """Apply a recommendation only if its schema passes worker validation."""
    if strategy == "schema":
        from ...chunk.schema import parse_chunking_schema

        try:
            parse_chunking_schema(parameters.get("json_schema"))
        except ValueError as exc:
            st.error(f"Cannot apply the recommended schema: {exc}")
            return False

    st.session_state.ai_recommended_strategy = strategy
    st.session_state.ai_recommended_params = parameters.copy()
    return True


def call_gemini_api(prompt: str, api_key: Optional[str], model_name: str) -> Optional[str]:
    """Call Gemini API for analysis."""
    try:
        if not api_key:
            st.error("❌ Gemini API key not configured.")
            return None

        genai = importlib.import_module("google.genai")
        client = genai.Client(api_key=api_key)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
        except Exception:
            response = client.models.generate_content(model=model_name, contents=prompt)

        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Gemini API error: {str(e)}")
        return None


def call_grok_api(prompt: str, api_key: Optional[str], model_name: str) -> Optional[str]:
    """Call Grok API for analysis."""
    try:
        if not api_key:
            st.error("❌ Grok API key not configured.")
            return None

        import requests

        api_url = "https://api.x.ai/v1/chat/completions"
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        response = requests.post(api_url, json=request_data, headers=headers)
        if response.status_code >= 400 and "response_format" in response.text.lower():
            request_data.pop("response_format", None)
            response = requests.post(api_url, json=request_data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        st.error(f"❌ Grok API error: {str(e)}")
        return None


def call_openai_api(prompt: str, api_key: Optional[str], model_name: str) -> Optional[str]:
    """Return completed answer text, retrying once on token exhaustion."""
    try:
        if not api_key:
            st.error("❌ OpenAI API key not configured.")
            return None

        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(api_key=api_key)
        request = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            **openai_sampling_options(model_name, temperature=0.1),
        }

        initial_budget, retry_budget = 1000, 8000  # Includes reasoning and visible output.
        for budget in (initial_budget, retry_budget):
            request["max_completion_tokens"] = budget
            try:
                response = client.chat.completions.create(**request)
            except openai_module.BadRequestError as error:
                if (
                    "response_format" not in request
                    or error.param != "response_format"
                    or error.code not in {"unsupported_parameter", "unsupported_value"}
                ):
                    raise
                request.pop("response_format")
                response = client.chat.completions.create(**request)

            if not response.choices:
                st.warning("OpenAI returned no completion choices for chunking analysis.")
                return None

            choice = response.choices[0]
            if choice.message.refusal:
                st.warning("OpenAI declined to answer the chunking analysis request.")
                return None
            if choice.finish_reason == "length":
                continue  # Discard partial JSON; the next attempt starts fresh.
            if choice.finish_reason != "stop":
                st.warning(f"OpenAI did not return a completed chunking answer ({choice.finish_reason}).")
                return None

            text = (choice.message.content or "").strip()
            if not text:
                st.warning("OpenAI completed the chunking analysis but returned no answer text.")
                return None
            return text

        st.warning(f"OpenAI reached the {retry_budget:,}-token retry limit without completing its answer.")
        return None
    except Exception as e:
        st.error(f"❌ OpenAI API error: {str(e)}")
        return None


def call_ollama_api(prompt: str, model_name: str) -> Optional[str]:
    """Call Ollama API for analysis."""
    try:
        import requests

        api_url = "http://localhost:11434/api/generate"
        request_data = {"model": model_name, "prompt": prompt, "stream": False}

        response = requests.post(api_url, json=request_data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("response", "").strip()
    except Exception as e:
        st.error(f"❌ Ollama API error: {str(e)}")
        return None
