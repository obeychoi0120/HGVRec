"""Current compact scene payloads and atomic writes from execution records."""

from pathlib import Path

from artifact_io import atomic_write_jsonl, read_jsonl
from extraction.graph_warnings import (
    DESCRIPTION_WARNING_TAGS, PARSE_FAILED, warning_tags,
)
from extraction.semantic_graph import parse_or_repair_graph, graph_semantic_warnings
from extraction.structured_output import (
    GRAPH_JSON_SCHEMA, OutputValidationError, normalize_graph_fields, normalize_graph_references,
)


class ExtractionStepError(RuntimeError):
    pass


class UnsupportedArtifactError(ExtractionStepError):
    """A stored scene row lacks fields the pipeline reads."""


def validate_tokens(value):
    """Output token counts come from the backend, never from text-length estimates."""
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("tokens must be a nonnegative integer or null")
    return value


def qwen_output_tokens(event):
    value = (event or {}).get("output_tokens")
    return value if type(value) is int and value >= 0 else None


def gemini_output_tokens(diagnostics):
    usage = (diagnostics or {}).get("usage_metadata") or {}
    value = usage.get("candidates_token_count", usage.get("candidatesTokenCount"))
    return value if type(value) is int and value >= 0 else None

GRAPH_FIELDS = ("content_id", "scene_idx", "tokens", "warning", "scene_graph")
DESCRIPTION_FIELDS = ("content_id", "scene_idx", "tokens", "description")
# Description rows carry "warning" only when tagged; untagged rows keep the older shape.
TAGGED_DESCRIPTION_FIELDS = (*DESCRIPTION_FIELDS, "warning")
FAILURE_FIELDS = ("content_id", "scene_idx", "tokens", "error", "raw_output")


def validate_scene_payload(path, row):
    """Check the fields downstream steps read; extra fields are allowed."""
    path = Path(path)
    fields = FAILURE_FIELDS if "error" in row else (
        GRAPH_FIELDS if "scene_graph" in row
        else TAGGED_DESCRIPTION_FIELDS if "warning" in row else DESCRIPTION_FIELDS
    )
    if not set(fields) <= set(row):
        raise UnsupportedArtifactError(f"missing scene fields {sorted(set(fields) - set(row))}: {path}")
    if row["content_id"] != path.stem or type(row["scene_idx"]) is not int or row["scene_idx"] < 0:
        raise ValueError(f"invalid scene identity: {path}")
    validate_tokens(row["tokens"])
    if "error" in row:
        if not isinstance(row["error"], str) or not row["error"].strip():
            raise ValueError(f"invalid scene error: {path}")
        if not isinstance(row["raw_output"], str):
            raise ValueError(f"invalid failed scene output: {path}")
    elif "scene_graph" in row:
        warning_tags(row["warning"])
        graph = row["scene_graph"]
        if isinstance(graph, dict):
            try:
                graph, _ = normalize_graph_fields(graph)
            except OutputValidationError as exc:
                raise UnsupportedArtifactError(f"unsupported Action fields: {path}: {exc}") from exc
            if set(graph) != set(GRAPH_JSON_SCHEMA["properties"]):
                raise UnsupportedArtifactError(f"unsupported Graph fields; expected Context/Entities/Actions: {path}")
            if isinstance(graph["actions"], list) and any(
                isinstance(action, dict) and set(action) != set(GRAPH_JSON_SCHEMA["properties"]["actions"]["items"]["properties"])
                for action in graph["actions"]
            ):
                raise UnsupportedArtifactError(f"unsupported Action fields; receiver-free roles and place required: {path}")
        elif not isinstance(graph, str) or not graph.strip():
            raise ValueError(f"invalid graph observation: {path}")
    else:
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError(f"invalid description observation: {path}")
        if "warning" in row and not set(warning_tags(row["warning"])) <= set(DESCRIPTION_WARNING_TAGS):
            raise ValueError(f"invalid description warning: {path}")


def _graph_warning(record, value):
    if "warning" in record:
        return warning_tags(record["warning"])
    if "semantic_warnings" in record:
        return warning_tags(record["semantic_warnings"])
    if isinstance(value, str):
        parsed = parse_or_repair_graph(value)
        return list(parsed.warning)
    return graph_semantic_warnings(value)


def recover_scene_graph(graph, warnings):
    """Read a stored Graph; stored tags are final, untagged rows get the current checks."""
    warnings = warning_tags(warnings)
    if warnings:
        return graph, warnings
    if isinstance(graph, str):
        parsed = parse_or_repair_graph(graph)
        return (
            graph if parsed.graph is None else parsed.graph,
            list(parsed.warning),
        )
    try:
        normalized, _ = normalize_graph_references(graph)
    except OutputValidationError as exc:
        return graph, exc.tags
    return normalized, []


def _payload(path, record):
    if "error" in record:
        payload = {key: record[key] for key in FAILURE_FIELDS}
        validate_scene_payload(path, payload)
        return payload
    fields = set(record) & {"description", "graph", "scene_graph"}
    if len(fields) != 1 or "raw_response" in record:
        raise UnsupportedArtifactError(f"unsupported scene execution record: {path}")
    field = fields.pop()
    payload = {
        "content_id": record.get("content_id", path.stem),
        "scene_idx": record.get("scene_idx"),
        "tokens": validate_tokens(record.get("tokens")),
    }
    if field == "description":
        payload["description"] = record[field]
        if record.get("semantic_warnings"):
            payload["warning"] = warning_tags(record["semantic_warnings"])
    else:
        try:
            graph, _ = normalize_graph_fields(record[field])
        except OutputValidationError as exc:
            raise UnsupportedArtifactError(f"unsupported Action fields: {path}: {exc}") from exc
        payload.update(warning=_graph_warning(record, graph), scene_graph=graph)
    validate_scene_payload(path, payload)
    fields = (GRAPH_FIELDS if "scene_graph" in payload
              else TAGGED_DESCRIPTION_FIELDS if "warning" in payload else DESCRIPTION_FIELDS)
    return {key: payload[key] for key in fields}


def read_scene_records(path, *, include_failures=False):
    """Adapt current persisted fields to the current in-memory execution contract."""
    path = Path(path)
    rows = read_jsonl(path)
    for row in rows:
        validate_scene_payload(path, row)
    if len({row["scene_idx"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate scene index: {path}")
    records = []
    for row in sorted(rows, key=lambda row: row["scene_idx"]):
        if "error" in row:
            if include_failures:
                records.append(row)
            continue
        common = {"scene_idx": row["scene_idx"], "keyframes": [], "tokens": row["tokens"]}
        if "description" in row:
            record = {
                "schema_version": "scene-description/v2", "content_id": path.stem,
                **common, "description": row["description"],
                "semantic_warnings": warning_tags(row.get("warning", [])),
            }
        else:
            graph, warnings = recover_scene_graph(row["scene_graph"], row["warning"])
            record = {
                **common, "graph": graph,
                "parse_mode": "text" if isinstance(graph, str) else (
                    "repaired" if graph != row["scene_graph"] else "unknown"
                ),
                "semantic_warnings": warnings,
            }
        records.append(record)
    return records


def reclassify_scene_row(row):
    """Apply the current parser to one stored row; returns the row unchanged if current.

    Graph text stored as PARSE_FAILED is parsed again with the current parser.
    """
    if "error" in row or "scene_graph" not in row:
        return row
    tags, graph = warning_tags(row["warning"]), row["scene_graph"]
    if tags == [PARSE_FAILED] and isinstance(graph, str):
        parsed = parse_or_repair_graph(graph)
        if parsed.graph is not None and parsed.error is None:
            return {**row, "warning": list(parsed.warning), "scene_graph": parsed.graph}
    elif not tags:
        tags = recover_scene_graph(graph, tags)[1]
    return row if tags == row["warning"] else {**row, "warning": tags}


def reclassify_scene_file(path):
    """Rewrite a stored scene file only when the current rules change some row."""
    path = Path(path)
    rows = read_jsonl(path)
    for row in rows:
        validate_scene_payload(path, row)
    current = [reclassify_scene_row(row) for row in rows]
    if current == rows:
        return False
    for row in current:
        validate_scene_payload(path, row)
    atomic_write_jsonl(path, current, durable=True, sort_keys=False)
    return True


def write_scene_records(path, records):
    """Validate all rows before publishing; never migrate existing artifacts."""
    path = Path(path)
    payload = sorted((_payload(path, row) for row in records), key=lambda row: row["scene_idx"])
    if len({row["scene_idx"] for row in payload}) != len(payload):
        raise ValueError(f"duplicate scene index: {path}")
    if payload:
        atomic_write_jsonl(path, payload, durable=True, sort_keys=False)
    else:
        path.unlink(missing_ok=True)


def ensure_generation_identity(directory, provenance, *, force=False):
    """Record the prompt/model settings of the first generation; never blocks resume."""
    from artifact_io import atomic_write_json

    directory = Path(directory)
    identity_path = directory / "generation_identity.json"
    identity = provenance
    directory.mkdir(parents=True, exist_ok=True)
    if force:
        # A new identity must never label old outputs after an interrupted regeneration.
        for path in [*directory.glob("*.jsonl"), *(directory / "failures").glob("*.jsonl")]:
            path.unlink()
    if force or not identity_path.exists():
        atomic_write_json(identity_path, {"identity": identity}, durable=True)
