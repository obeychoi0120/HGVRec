"""Recommendation definitions from rec_experiments_config.yaml, loaded once per invocation."""

from pathlib import Path
import re

import yaml

CONFIG_PATH = Path("rec_experiments_config.yaml")
CONFIG_KEY = "_rec_experiments"
# Internal module names in canonical application order.
MODULES = ("role_fusion", "scene_attention", "activity_hypergraph")
# Names written in rec_experiments_config.yaml.
MODULE_ALIASES = {
    "REAF": "role_fusion",
    "GNN_ATTN": "scene_attention",
    "ASHA": "activity_hypergraph",
}


MODULE_DISPLAY_NAMES = {internal: name for name, internal in MODULE_ALIASES.items()}


def display_definition(definition):
    """Copy of a recorded definition with config module names, for logs and reports."""
    return {**definition, "modules": [MODULE_DISPLAY_NAMES.get(m, m) for m in definition["modules"]]}


def parse_experiments(value):
    from arm_registry import generation_registry

    if not isinstance(value, dict):
        raise ValueError("recommendation config must be a mapping with an experiments list")
    defaults = value.get("defaults") or {}
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    sources = generation_registry()
    names, result = set(), []
    for row in experiments:
        if not isinstance(row, dict):
            raise ValueError("each experiment must be a mapping")
        name = row.get("label")
        # Labels name result directories: only keep characters that are safe in a path.
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+.-]*", name) is None:
            raise ValueError("label must start with a letter or digit and contain only "
                             "letters, digits, _, +, ., -")
        if name.lower() in names:
            raise ValueError(f"duplicate experiment label (case-insensitive): {name}")
        names.add(name.lower())
        source = row.get("source")
        if not isinstance(source, str) or source not in {"metadata", *sources}:
            raise ValueError(f"{name}: source must explicitly select metadata or a registered extraction source")
        modules = row.get("modules", [])
        if not isinstance(modules, list) or any(not isinstance(m, str) for m in modules):
            raise ValueError(f"{name}: modules must be a list of module names")
        modules = [MODULE_ALIASES.get(m, m) for m in modules]
        if len(set(modules)) != len(modules) or set(modules) - set(MODULES):
            raise ValueError(f"{name}: modules must be unique supported module names "
                             f"({', '.join(MODULE_ALIASES)})")
        if modules and (source == "metadata" or sources[source].representation != "graph"):
            raise ValueError(f"{name}: modules are graph-only")
        readout = row.get("readout", defaults.get("readout", "mean"))
        resolved = {"label": name, "source": source,
                    "modules": [m for m in MODULES if m in modules], "readout": readout}
        if "segments" in row or "segments_seed" in row:
            resolved["segments"] = parse_segments(name, row, modules)
        result.append(resolved)
    return {"experiments": result}


def parse_segments(name, row, modules):
    from validation.segment_variants import VARIANTS

    if "activity_hypergraph" not in modules:
        raise ValueError(f"{name}: segments requires the ASHA module")
    variant = row.get("segments")
    if variant not in VARIANTS:
        raise ValueError(f"{name}: segments must be one of {', '.join(VARIANTS)}")
    seed = row.get("segments_seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError(f"{name}: segments_seed must be a non-negative integer")
    return {"variant": variant, "seed": seed}


def load_experiments(context):
    """Freeze a resolved snapshot in the context; spawned workers inherit it."""
    if CONFIG_KEY not in context.config:
        path = context.root / CONFIG_PATH
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            context.config[CONFIG_KEY] = parse_experiments(value)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"invalid recommendation config {path}: {exc}") from exc
    return context.config[CONFIG_KEY]


def experiment_definition(arm):
    from arm_registry import feature_source

    modules = (["role_fusion"] if arm.role_fusion else [])
    if "scene_attention" in arm.extensions:
        modules.append("scene_attention")
    if "segments" in arm.extensions:
        modules.append("activity_hypergraph")
    definition = {"label": arm.name, "source": feature_source(arm), "modules": modules,
                  "readout": arm.aggregation}
    if arm.segments is not None:
        definition["segments"] = {"variant": arm.segments[0], "seed": arm.segments[1]}
    return definition
