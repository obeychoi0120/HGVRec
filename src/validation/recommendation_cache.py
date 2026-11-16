"""Cross-run reuse of completed recommendation combinations."""

from artifact_io import fingerprint, read_json, write_json
from validation.shared_cache import SharedCache

RESULT_FILES = ["sasrec.pt", "training.json", "per_event_metrics.jsonl", "catalog_vectors.npy",
                "complete.json"]
# Keys that only describe where a result lives, not how it was produced.
LOCAL_KEYS = {"run_id", "arm"}


def cache_for(context, identity):
    key = fingerprint({k: v for k, v in identity.items() if k not in LOCAL_KEYS})
    return SharedCache(context, "recommendations", key)


def restore(context, directory, identity):
    cache = cache_for(context, identity)
    if not cache.restore(directory):
        return False
    complete = read_json(directory / "complete.json")
    complete["reused_from"] = {"source_run_id": complete.get("identity", {}).get("run_id"),
                               "cache_key": cache.key}
    complete["identity"] = identity
    write_json(directory / "complete.json", complete)
    return True


def publish(context, directory):
    """Share a finished combination under the identity it was trained with."""
    if (directory / "complete.json").is_file():
        identity = read_json(directory / "complete.json")["identity"]
        cache_for(context, identity).publish(directory, RESULT_FILES,
                                             origin={"run_id": context.run_id})
