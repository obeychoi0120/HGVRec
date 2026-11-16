from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.evidence import build_scene_evidence
from extraction.progress import InferenceProgress
from extraction.scene_storage import ExtractionStepError, write_scene_records
from pipeline_runtime import RunContext
from artifact_io import read_jsonl


def video_names(catalog: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in catalog:
        content_id = str(row["content_id"])
        source = str(row.get("source_video_path") or "").strip()
        names[content_id] = Path(source.replace("\\", "/")).name if source else f"{content_id}.mp4"
    return names


def write_progress(progress: tqdm | InferenceProgress, message: str) -> None:
    if isinstance(progress, InferenceProgress):
        progress.write_log(message)
    else:
        tqdm.write(message, file=progress.fp)


def write_scene_results(scene_path: Path, records: list[dict[str, Any]]) -> None:
    """Publish completed scene results without a recovery journal."""
    records.sort(key=lambda row: int(row["scene_idx"]))
    write_scene_records(scene_path, records)


def video_name_map(context: RunContext) -> dict[str, str]:
    return video_names(read_jsonl(context.cohort_dir / "catalog.jsonl"))


def scene_generation_rows(
    visual: dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    generation: dict | None = None,
    scenes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if scenes is None:
        scenes = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in build_scene_evidence(
        scenes, visual["frames_dir"], visual["timestamp_json"], prepared=True
    ):
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise ExtractionStepError(
                f"{visual['content_id']} scene {scene_idx} has "
                f"{len(image_paths)} of {len(keyframes)} keyframes"
            )
        task_id = f"{visual['content_id']}:{scene_idx}"
        rows.append(
            {
                "task": QwenGenerationTask(
                    task_id=task_id,
                    image_paths=tuple(image_paths),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **({"repetition_penalty": repetition_penalty} | (generation or {})),
                ),
                "scene_idx": scene_idx,
                "scene_start_seconds": scene["scene_start_seconds"],
                "scene_end_seconds": scene["scene_end_seconds"],
                "keyframes": keyframes,
                "image_paths": image_paths,
            }
        )
    if not rows:
        raise ExtractionStepError(f"{visual['content_id']} has no scenes")
    return rows


def result(
    stage: str,
    *,
    content_count: int,
    failure_count: int = 0,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "content_count": content_count,
        "failure_count": failure_count,
    }


def visual_rows(context: RunContext) -> list[dict[str, Any]]:
    cohort = context.require_ready_cohort()
    from visual_sampling import timestamp_filename

    sampling = context.config["extraction"]["visual_evidence"]
    rows: list[dict[str, Any]] = []
    for item in cohort["catalog"]:
        content_id = str(item["content_id"])
        frames_dir = context.keyframes_dir / content_id
        timestamp = (
            context.source_assets_dir
            / content_id
            / timestamp_filename(sampling["scene_duration"], sampling["num_keyframes"])
        )
        rows.append(
            {
                "content_id": content_id,
                "item_id": str(item["item_id"]),
                "frames_dir": str(frames_dir),
                "timestamp_json": str(timestamp),
            }
        )
    if not rows:
        raise ExtractionStepError(f"empty cohort catalog: {context.cohort_dir}")
    return rows
