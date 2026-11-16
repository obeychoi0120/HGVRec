import pytest

from artifact_io import read_json
from helpers import make_context, prepare_inputs, result_identity, selected_table, train_inputs
from validation.leave_one_out_recommendation import combination_complete
from validation.selection import training_signature
from validation.shared_cache import SharedCache
from validation.steps import validation_config


def _check_training_signature_follows_settings(context, monkeypatch):
    prepare_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    config = validation_config(context)
    original = training_signature(context, cohort, config)
    changed = config.model_copy(deep=True)
    changed.model.learning_rate *= 2
    assert training_signature(context, cohort, changed) != original
    reseeded = config.model_copy(deep=True)
    reseeded.model.seeds = [1, 2, 3]
    # Seeds are keyed per combination, not in the shared training signature.
    assert training_signature(context, cohort, reseeded) == original


def _check_shared_cache_roundtrip(context):
    source = context.root / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"features")
    cache = SharedCache(context, "regression", "key")
    assert not cache.restore(context.root / "never")
    cache.publish(source, ["data.bin"], origin={"run_id": "original"})
    target = context.root / "restored"
    assert cache.restore(target)
    assert (target / "data.bin").read_bytes() == b"features"
    assert not (target / "cache.json").exists()


def test_training_signature_and_shared_cache(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        _check_training_signature_follows_settings(make_context(tmp_path / "0" / "repo"), patch)
    _check_shared_cache_roundtrip(make_context(tmp_path / "1" / "repo"))


@pytest.mark.torch
def test_resume_uses_completion_marker_and_force_retrains(context, monkeypatch):
    from validation.steps import run_recommendation

    train_inputs(context, monkeypatch)
    cohort, _, _ = selected_table(context)
    directory = context.recommendations_dir / "seed_42" / "base"
    assert combination_complete(directory)
    assert read_json(directory / "complete.json")["identity"] == result_identity(context, cohort)

    calls = []
    monkeypatch.setattr("validation.leave_one_out_recommendation.run_combination",
                        lambda *args, **kwargs: calls.append(args[4]["arm"]))
    # Changing settings does not block or invalidate finished results; they are reused.
    context.config["validation"]["model"]["learning_rate"] *= 2
    assert run_recommendation(context, target=["base"])["completed"] == 0
    # A missing marker trains again: the changed settings have no shared result to restore.
    (directory / "complete.json").unlink()
    run_recommendation(context, target=["base"])
    assert calls == ["base"]
    calls.clear()
    # The original settings restore the deleted marker's result from the shared cache.
    context.config["validation"]["model"]["learning_rate"] /= 2
    assert run_recommendation(context, target=["base"])["completed"] == 0 and not calls
    assert combination_complete(directory)
    run_recommendation(context, target=["base"], force=True)
    assert calls == ["base"] * 3
