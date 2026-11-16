"""Embedding state shared by recommendation results and diagnosis reports."""

from artifact_io import fingerprint, read_json


def read_state(context, branch):
    from arm_registry import feature_source, registry

    source = feature_source(registry(context.config)[branch])
    root = context.representations_dir / source
    manifest = read_json(root / "manifest.json")
    return {
        "input_hash": manifest["input_hash"],
        "recommendation_hash": fingerprint(manifest),
        "sources": [{"source": source, "input_hash": manifest["input_hash"],
                     "version": manifest.get("version")}],
        "statistics": read_json(root / "statistics.json"),
    }


def recommendation_identity(context, branch):
    """What a trained result depends on besides seed and split; used as the shared-cache key."""
    from arm_registry import registry
    from rec_experiments import experiment_definition

    arm = registry(context.config)[branch]
    definition = experiment_definition(arm)
    return {
        "experiment_definition": definition,
        "representation_mode": arm.representation,
        "embedding_hash": read_state(context, branch)["input_hash"],
    }
