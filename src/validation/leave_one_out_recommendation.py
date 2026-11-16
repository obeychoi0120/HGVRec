"""Independent seed/arm best-checkpoint evaluation and atomic resume."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from artifact_io import write_json
from validation.leave_one_out_data import metrics_from_rank
from validation.early_stopping import EarlyStopping
from validation.model import require_torch, save_checkpoint, seed_everything, torch
from arm_registry import resolve_target_arms
from validation.leave_one_out_data import EventTable, PROTOCOL
from validation.selection import load_validation_cohort, training_signature
from validation.leave_one_out_execution import (
    EXECUTION_VERSION,
    execution_for,
    masked_ranks,
    negative_mask,
)


def _optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.model.learning_rate, weight_decay=0.1)


SCHEMA = "sasrec-leave-one-out-combination/v2"


def phase_ids(table, split, phase):
    ids = np.asarray(split["phases"][phase]["event_ids"], dtype=np.int64)
    return ids[table.history_ends[ids] > 0]


def transition_loss(model, table, ids, probabilities, device):
    execution = execution_for(table)
    target_ids = table.targets[ids]
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=device)
    inputs = execution.inputs(ids, model.max_length, device)
    if hasattr(model, "prepare_items"):
        graph_ids = np.concatenate(
            [execution.padded(model.max_length)[ids].reshape(-1), target_ids]
        )
        model.prepare_items(graph_ids)
    users = model.user_vectors(inputs)
    candidates = model.item_vectors(targets)
    logits = users @ candidates.T
    logs = execution.log_probabilities(probabilities, target_ids, device, logits.dtype)
    logits = logits - logs[targets][None, :]
    masks = negative_mask(inputs, targets)
    logits = logits.masked_fill(masks, -1e4)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ids), device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite transition loss")
    return loss


def train_epoch(
    model,
    optimizer,
    table,
    ids,
    probabilities,
    rng,
    config,
    device,
):
    model.train()
    total = torch.zeros((), dtype=torch.float64, device=device)
    updates = 0
    order = rng.permutation(ids)
    for start in range(0, len(order), config.model.batch_size):
        batch = order[start : start + config.model.batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = transition_loss(model, table, batch, probabilities, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if hasattr(model, "clear_item_cache"):
            model.clear_item_cache()
        total += loss.detach().to(torch.float64) * len(batch)
        updates += 1
    if not len(ids):
        raise RuntimeError("empty training partition")
    return {
        "loss": float(total.cpu()) / len(ids),
        "positive_count": len(ids),
        "optimizer_updates": updates,
    }


def rank_batches(model, table, ids, config, device):
    """Share ranking between validation-only and full per-event evaluation."""
    model.eval()
    execution = execution_for(table)
    with torch.no_grad():
        catalog = model.catalog_vectors()
        for start in range(0, len(ids), config.model.batch_size):
            batch = ids[start : start + config.model.batch_size]
            inputs = execution.inputs(batch, model.max_length, device)
            users = model.user_vectors(inputs)
            scores = users @ catalog.T
            ranks = masked_ranks(scores, table, batch)
            ranks = ranks.cpu().tolist()
            yield batch, ranks


def validation_hr10(model, table, ids, config, device):
    return sum(
        float(rank <= 10)
        for _, ranks in rank_batches(model, table, ids, config, device)
        for rank in ranks
    ) / len(ids)


def evaluate(model, table, ids, config, device):
    """Frozen parameters, cached catalog vectors, full preceding sequence mask."""
    for batch, ranks in rank_batches(model, table, ids, config, device):
        for event, rank in zip(batch, ranks, strict=True):
            yield {
                **table.rows[int(event)],
                "rank": rank,
                "candidate_count": len(table.items),
                **metrics_from_rank(rank, config.evaluation.cutoffs),
            }


def combination_dir(context, seed, arm):
    return context.recommendations_dir / f"seed_{seed}" / arm.lower()


def combination_complete(directory):
    """Resume rule: a combination is done once its completion marker exists."""
    return (directory / "complete.json").is_file()


def run_combination(
    context, config, table, split, identity, branch, prepared, device,
):
    ids, probabilities, frequencies = prepared
    seed, arm = (identity[key] for key in ("seed", "arm"))
    directory = combination_dir(context, seed, arm)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "identity.json", identity)
    (directory / "complete.json").unlink(missing_ok=True)
    print(f"[Leave-one-out] seed={seed} {arm}: train device={device}", flush=True)
    started = time.monotonic()
    timings = dict.fromkeys(
        ("preparation", "train", "validation", "checkpoint_restore", "test"), 0.0
    )
    execution = execution_for(table)
    execution.padded(config.model.max_sequence_length)
    execution.log_probabilities(
        probabilities["train"], table.targets[ids["train"]], device, torch.float32
    )
    timings["preparation"] = time.monotonic() - started
    def create_model():
        from validation.graph_model import new_graph_model

        return new_graph_model(context, config, branch, device)

    seed_everything(seed, deterministic=config.model.deterministic)
    rng = np.random.default_rng(seed)
    model = create_model()
    optimizer = _optimizer(model, config)
    train = []
    stopping = EarlyStopping(config.model.patience, getattr(config.model, "min_delta", 0.0))
    for epoch in range(1, config.model.max_epochs + 1):
        phase_started = time.monotonic()
        record = train_epoch(
            model,
            optimizer,
            table,
            ids["train"],
            probabilities["train"],
            rng,
            config,
            device,
        )
        train_seconds = time.monotonic() - phase_started
        timings["train"] += train_seconds
        phase_started = time.monotonic()
        score = validation_hr10(
            model, table, ids["validation"], config, device
        )
        valid_seconds = time.monotonic() - phase_started
        timings["validation"] += valid_seconds
        train.append({"epoch": epoch, **record, "validation_hr10": score})
        print(
            f"[Leave-one-out] {arm} seed={seed} epoch={epoch} valid={score:.6f} "
            f"train={train_seconds:.2f}s valid_time={valid_seconds:.2f}s "
            f"train_events/s={len(ids['train']) / max(train_seconds, 1e-9):.1f}",
            flush=True,
        )
        previous_best = stopping.best_epoch
        stopped = stopping.update(epoch, score)
        if stopping.best_epoch != previous_best:
            save_checkpoint(
                directory / "sasrec.pt", model,
                {**identity, "best_epoch": epoch, "catalog_size": len(table.items),
                 "checkpoint_policy": "best_validation_hr10"},
            )
        if stopped:
            break
    best_epoch = stopping.best_epoch
    phase_started = time.monotonic()
    checkpoint = torch.load(directory / "sasrec.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    if hasattr(model, "clear_item_cache"):
        model.clear_item_cache()
    timings["checkpoint_restore"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    print(f"[Leave-one-out] {arm} seed={seed} test device={device}", flush=True)
    temporary = directory / "per_event_metrics.jsonl.tmp"
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in evaluate(model, table, ids["test"], config, device):
            row.update(identity)
            row["schema_version"] = "sasrec-leave-one-out-per-event/v2"
            row["train_item_frequency"] = int(frequencies[table.targets[row["event_id"]]])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if count != len(ids["test"]):
        raise RuntimeError("incomplete evaluation")
    temporary.replace(directory / "per_event_metrics.jsonl")
    timings["test"] = time.monotonic() - phase_started
    print(f"[Leave-one-out] {arm} seed={seed} test={timings['test']:.2f}s", flush=True)
    metadata = {
        **identity,
        "best_epoch": best_epoch,
        "checkpoint_policy": "best_validation_hr10",
        "catalog_size": len(table.items),
        "training_settings": {
            "deterministic": config.model.deterministic,
            "learning_rate": config.model.learning_rate,
            "batch_size": config.model.batch_size,
            "patience": config.model.patience,
            "min_delta": getattr(config.model, "min_delta", 0.0),
        },
    }
    from validation.graph_model import parameter_report

    metadata["parameters"] = parameter_report(model, branch)
    with torch.no_grad():
        np.save(directory / "catalog_vectors.npy", model.catalog_vectors().cpu().numpy())
    save_checkpoint(directory / "sasrec.pt", model, metadata)
    write_json(
        directory / "training.json",
        {
            "schema_version": SCHEMA,
            **metadata,
            "split": split,
            "train": train,
            "untrained_test_target_fraction": float(
                np.mean(frequencies[table.targets[ids["test"]]] == 0)
            ),
            "elapsed_seconds": time.monotonic() - started,
            "execution": {
                "version": EXECUTION_VERSION,
                "seconds": timings,
                **(
                    {
                        "graph_training": model.execution_report(),
                    }
                    if hasattr(model, "execution_report")
                    else {}
                ),
            },
            "device": str(device),
            "environment": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )
    write_json(
        directory / "complete.json",
        {"schema_version": SCHEMA, "identity": identity, "event_count": count},
    )
    # Publish each completed unit, including in spawned workers, so interruption
    # of a later combination does not withhold already finished shared results.
    from validation import recommendation_cache

    recommendation_cache.publish(context, directory)


def prepare_split(table, split):
    ids = {phase: phase_ids(table, split, phase) for phase in split["phases"]}
    counts = np.bincount(table.targets[ids["train"]], minlength=len(table.items) + 1)
    probabilities = {"train": counts / counts.sum()}
    raw_train = np.asarray(split["phases"]["train"]["event_ids"], dtype=np.int64)
    frequencies = np.bincount(table.targets[raw_train], minlength=len(table.items) + 1)
    return ids, probabilities, frequencies


def worker_devices(workers_per_gpu):
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not available:
        if workers_per_gpu == 1:
            return ["cpu"]
        raise ValueError("workers_per_gpu requires visible CUDA devices")
    # Logical indices respect CUDA_VISIBLE_DEVICES, including UUID masks.
    # Round-robin order spreads a small pending set across visible GPUs first.
    return [f"cuda:{i}" for _ in range(workers_per_gpu) for i in range(available)]


def run_leave_one_out(
    context, *, force=False, workers_per_gpu=1, target=None,
):
    from validation.config import validation_config
    from validation.representation_provenance import recommendation_identity

    arms = resolve_target_arms(target, config=context.config)
    from validation import recommendation_cache

    config = validation_config(context)
    cohort = load_validation_cohort(context)
    table = EventTable(cohort["events"])
    training_input_hash = training_signature(context, cohort, config)
    completed = skipped = shared = 0
    jobs = []
    combinations = []
    for split in cohort["plan"]["splits"]:
        for seed in config.model.seeds:
            for arm, branch in arms.items():
                identity = {
                    "run_id": context.run_id,
                    "protocol": PROTOCOL,
                    "split_hash": split["split_hash"],
                    "seed": seed,
                    "arm": arm,
                    "training_input_hash": training_input_hash,
                    "deterministic": config.model.deterministic,
                    **recommendation_identity(context, branch),
                }
                directory = combination_dir(context, seed, arm)
                combinations.append((directory, identity, branch, split))
    for directory, identity, branch, split in combinations:
        if not force and combination_complete(directory):
            skipped += 1
        elif not force and recommendation_cache.restore(context, directory, identity):
            shared += 1
            skipped += 1
        else:
            jobs.append((split, identity, branch))
    devices = []
    if jobs:
        require_torch()
        devices = worker_devices(workers_per_gpu)
    total = skipped + len(jobs)
    print(
        f"[Leave-one-out] pending={len(jobs)} reused={skipped} "
        f"workers={min(len(devices), len(jobs))} devices={','.join(dict.fromkeys(devices))}",
        flush=True,
    )
    with tqdm(
        total=total, initial=skipped, desc="Leave-one-out recommendation", unit="run", file=sys.stdout
    ) as progress:
        progress.set_postfix(reused=skipped)
        if jobs and len(devices) > 1:
            from validation.leave_one_out_workers import run_parallel

            completed = run_parallel(context, jobs, devices, progress)
        elif jobs:
            previous_split, prepared = None, None
            device = torch.device(devices[0])
            for split, identity, branch in jobs:
                if split["split_hash"] != previous_split:
                    prepared = prepare_split(table, split)
                    previous_split = split["split_hash"]
                progress.set_postfix(
                    seed=identity["seed"], arm=identity["arm"], reused=skipped
                )
                run_combination(
                    context,
                    config,
                    table,
                    split,
                    identity,
                    branch,
                    prepared,
                    device,
                )
                completed += 1
                progress.update(1)
    for directory, _, _, _ in combinations:
        recommendation_cache.publish(context, directory)
    write_json(
        context.recommendations_dir / "reuse.json",
        {
            "run_id": context.run_id,
            "local": skipped - shared,
            "shared": shared,
            "generated": completed,
        },
    )
    print(f"[Leave-one-out] completed={completed} skipped={skipped}", flush=True)
    return {"stage": "run-recommendation", "completed": completed, "skipped": skipped}
