from __future__ import annotations

from typing import Any

from .errors import RouterError


BLOCKED_ROUTING_KEYS = {
    "facet",
    "facets",
    "primaryfacet",
    "activefacet",
    "activefacets",
    "expectedfacet",
    "expectedfacets",
    "routinglabel",
    "routinglabels",
}


def compact_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def first_value(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return compact_whitespace(value)
    if isinstance(value, list):
        return "; ".join(
            rendered for item in value if (rendered := render_value(item))
        )
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if str(key).replace("_", "").casefold() in BLOCKED_ROUTING_KEYS:
                continue
            rendered = render_value(item)
            if rendered:
                parts.append(f"{key}={rendered}")
        return "; ".join(parts)
    return compact_whitespace(value)


def append_field(lines: list[str], label: str, value: Any) -> None:
    rendered = render_value(value)
    if rendered:
        lines.append(f"{label}: {rendered}")


def agent_event_to_text(
    event: Any,
    task_context: Any = None,
) -> str:
    """Build the semantic LSR view of one agent transition.

    The projection intentionally ignores facet labels, expected labels and model
    targets.  It uses only state that is also available before inference.
    """

    if not isinstance(event, dict):
        raise RouterError("agentEvent must be an object.")
    context = task_context if isinstance(task_context, dict) else {}
    lines: list[str] = []

    append_field(lines, "Route scope", first_value(event, "scope", "routeScope"))
    append_field(lines, "Agent phase", first_value(event, "phase", "agentPhase"))
    append_field(
        lines,
        "Task objective",
        first_value(event, "objective", "goal")
        or first_value(context, "objective", "goal"),
    )
    append_field(
        lines,
        "Current step",
        first_value(event, "currentStep", "current_step", "step"),
    )
    append_field(
        lines,
        "Requested operation",
        first_value(event, "requestedOperation", "requested_operation", "operation"),
    )
    append_field(
        lines,
        "Pending decision",
        first_value(event, "pendingDecision", "pending_decision"),
    )

    artifact = first_value(event, "artifact", "currentArtifact", "current_artifact")
    append_field(lines, "Artifact", artifact)
    observation = first_value(event, "observation", "latestObservation", "latest_observation")
    append_field(lines, "Latest observation", observation)

    constraints = first_value(event, "constraints", "activeConstraints", "active_constraints")
    if constraints is None:
        constraints = context.get("constraints")
    append_field(lines, "Constraints", constraints)
    append_field(
        lines,
        "Available tools",
        first_value(event, "availableTools", "available_tools")
        or first_value(context, "availableTools", "available_tools"),
    )
    append_field(
        lines,
        "Domain context",
        first_value(event, "domainContext", "domain_context")
        or first_value(context, "domainContext", "domain_context"),
    )
    append_field(
        lines,
        "Response language",
        first_value(event, "language", "responseLanguage", "response_language")
        or first_value(context, "language", "responseLanguage", "response_language"),
    )
    append_field(
        lines,
        "History digest",
        first_value(event, "historyDigest", "history_digest"),
    )

    if not lines:
        raise RouterError("agentEvent has no routable state fields.")
    return "\n".join(lines)


def messages_to_routing_text(messages: Any, index: int = 0) -> str:
    """Project prompt-side messages while excluding assistant targets.

    Canonical v3 records keep targets outside ``input.messages``.  Excluding
    assistant messages here also fixes target leakage for legacy SFT files.
    """

    if not isinstance(messages, list) or not messages:
        raise RouterError(f"messages item {index} must contain a non-empty messages list.")
    parts: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RouterError(f"messages item {index}.{message_index} must be an object.")
        role = compact_whitespace(message.get("role") or "")
        if role.casefold() == "assistant":
            continue
        content = compact_whitespace(message.get("content") or "")
        if content:
            parts.append(f"{role}: {content}" if role else content)
    if not parts:
        raise RouterError(f"messages item {index} has no prompt-side message content.")
    return "\n".join(parts)


def facet_record_to_routing_text(item: dict[str, Any], index: int = 0) -> str:
    """Derive the LSR input from the same canonical row used for facet training."""

    record_input = item.get("input")
    if not isinstance(record_input, dict):
        raise RouterError(f"Facet record {index} must contain an input object.")

    activation = item.get("activation") if isinstance(item.get("activation"), dict) else {}
    route_from = str(activation.get("routeFrom") or "").strip()
    content = compact_whitespace(record_input.get("content") or "")
    event = first_value(record_input, "agentEvent", "agent_event")
    task_context = first_value(record_input, "taskContext", "task_context")
    messages = record_input.get("messages")

    if route_from == "input.content":
        if not content:
            raise RouterError(f"Facet record {index} routes from missing input.content.")
        return content
    if route_from == "input.agentEvent":
        if event is None:
            raise RouterError(f"Facet record {index} routes from missing input.agentEvent.")
        return agent_event_to_text(event, task_context)
    if route_from == "input.messages":
        return messages_to_routing_text(messages, index)

    parts: list[str] = []
    if event is not None:
        parts.append(agent_event_to_text(event, task_context))

    if messages:
        parts.append(messages_to_routing_text(messages, index))

    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = part.strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
    if not unique:
        raise RouterError(f"Facet record {index} has no routable training input.")
    return "\n\n".join(unique)
