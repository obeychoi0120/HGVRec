"""scene_attention (GNN_ATTN): exchange math, video isolation, batching, init and identity compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arm_registry import registry
from helpers import PRE_SCENE_ATTENTION_EXPERIMENTS, experiment_identities
from rec_experiments import CONFIG_KEY, parse_experiments
from test_scene_extensions import make_model, synthetic_store
from validation.graph_batching import PackedGraphStore
from validation.graph_context import EXTENSION_MODELS
from validation.graph_model import SceneAttentionExchange
from validation.scene_attention import scene_groups, unit_rms

pytestmark = pytest.mark.torch

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "pre_scene_attention.json").read_text())
EXCHANGE = ("scene_attention",)


def ragged(counts=(3, 1, 4, 3, 2, 4), dimension=32, similarity_dim=24, seed=1):
    generator = torch.Generator().manual_seed(seed)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    scenes = torch.randn(offsets[-1], dimension, generator=generator)
    similarity = torch.randn(offsets[-1], similarity_dim, generator=generator)
    return offsets, scenes, similarity


def grouped(offsets):
    return [torch.as_tensor(value) for value in scene_groups(offsets).values()]


def exchange_one(module, scenes, similarity):
    """Reference: one video, written straight from the specification."""
    units = similarity / similarity.norm(dim=-1, keepdim=True).clamp_min(1e-8) * similarity.shape[-1] ** 0.5
    q, k = units @ module.query.t(), units @ module.key.t()
    weights = torch.softmax(q @ k.t() / module.query.shape[0] ** 0.5, dim=-1)
    return scenes + torch.relu((weights @ scenes) @ module.projection.t() + module.projection_bias)


def test_grouped_exchange_matches_per_video_reference():
    torch.manual_seed(0)
    module = SceneAttentionExchange(32, 24, 8)
    offsets, scenes, similarity = ragged()
    with torch.no_grad():
        actual = module(scenes, similarity, grouped(offsets))
        for start, stop in zip(offsets[:-1], offsets[1:]):
            expected = exchange_one(module, scenes[start:stop], similarity[start:stop])
            torch.testing.assert_close(actual[start:stop], expected, atol=1e-6, rtol=0)


def test_videos_never_mix():
    torch.manual_seed(0)
    module = SceneAttentionExchange(32, 24, 8)
    offsets, scenes, similarity = ragged()
    changed_scenes, changed_similarity = scenes.clone(), similarity.clone()
    changed_scenes[offsets[2]:offsets[3]] += 5.0  # video 2 only
    changed_similarity[offsets[2]:offsets[3]] *= -1.0
    with torch.no_grad():
        before = module(scenes, similarity, grouped(offsets))
        after = module(changed_scenes, changed_similarity, grouped(offsets))
    inside = torch.zeros(len(scenes), dtype=torch.bool)
    inside[offsets[2]:offsets[3]] = True
    assert torch.equal(before[~inside], after[~inside])
    assert not torch.allclose(before[inside], after[inside])


def test_permuting_scenes_permutes_updates_and_keeps_the_video_mean():
    torch.manual_seed(0)
    module = SceneAttentionExchange(32, 24, 8)
    offsets, scenes, similarity = ragged()
    start, stop = offsets[2], offsets[3]  # four scenes
    order = torch.arange(len(scenes))
    order[start:stop] = torch.tensor([3, 0, 2, 1]) + start
    with torch.no_grad():
        before = module(scenes, similarity, grouped(offsets))
        after = module(scenes[order], similarity[order], grouped(offsets))
    torch.testing.assert_close(after, before[order], atol=1e-6, rtol=0)
    torch.testing.assert_close(after[start:stop].mean(0), before[start:stop].mean(0), atol=1e-6, rtol=0)


def test_single_scene_video_is_a_residual_projection_and_trains():
    torch.manual_seed(0)
    module = SceneAttentionExchange(32, 24, 8)
    offsets, scenes, similarity = ragged(counts=(1, 2))
    output = module(scenes, similarity, grouped(offsets))
    expected = scenes[0] + torch.relu(module.projection @ scenes[0] + module.projection_bias)
    torch.testing.assert_close(output[0], expected, atol=1e-6, rtol=0)
    output.square().sum().backward()
    for name in ("query", "key", "projection", "projection_bias"):
        gradient = getattr(module, name).grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_scene_free_videos_keep_the_title_fallback():
    store = synthetic_store()
    reference, model = make_model(store, role_fusion=False), make_model(store, EXCHANGE, role_fusion=False)
    empty = np.flatnonzero(np.diff(np.asarray(store["video_offsets"])) == 0)
    assert len(empty)
    with torch.no_grad():
        expected, actual = reference.catalog_vectors(), model.catalog_vectors()
    assert torch.equal(actual[empty], expected[empty])
    assert not torch.allclose(actual, expected)


def test_parameter_count():
    module = SceneAttentionExchange(512, 384, 128)
    assert sum(p.numel() for p in module.parameters()) == 2 * 128 * 384 + 512 * 512 + 512 == 360_960
    store = synthetic_store()
    for role_fusion in (False, True):
        plain, exchanged = (sum(p.numel() for p in make_model(store, e, role_fusion=role_fusion).parameters())
                            for e in ((), EXCHANGE))
        assert exchanged - plain == 2 * 8 * 24 + 32 * 32 + 32


@pytest.mark.parametrize("role_fusion", [False, True])
def test_shared_parameters_start_bitwise_equal_and_rng_is_untouched(role_fusion):
    store = synthetic_store()
    reference = make_model(store, role_fusion=role_fusion)
    expected_rng = torch.rand(3)
    model = make_model(store, EXCHANGE, role_fusion=role_fusion)
    assert torch.equal(torch.rand(3), expected_rng)
    state = model.state_dict()
    assert set(state) - set(reference.state_dict()) == {
        f"graph_encoder.scene_attention.{n}" for n in ("query", "key", "projection", "projection_bias")}
    assert all(torch.equal(value, state[name]) for name, value in reference.state_dict().items())


def test_initial_logits_have_std_about_half():
    torch.manual_seed(0)
    module = SceneAttentionExchange(512, 384, 128)
    units = unit_rms(torch.randn(4096, 384))
    with torch.no_grad():
        logits = (units @ module.query.t() * (units @ module.key.t())).sum(-1) / 128 ** 0.5
        # Off-diagonal pairs as well as each scene with itself.
        cross = (units[:2048] @ module.query.t() * (units[2048:] @ module.key.t())).sum(-1) / 128 ** 0.5
    assert 0.35 < float(logits.std()) < 0.65
    assert 0.35 < float(cross.std()) < 0.65
    assert module.query.std().item() == pytest.approx((2 * 384) ** -0.5, rel=0.05)


def test_title_changes_values_but_not_attention_weights():
    store = synthetic_store()
    model = make_model(store, EXCHANGE, role_fusion=False)
    exchange = model.graph_encoder.scene_attention
    recorded, original = [], exchange.attention

    def record(similarity):
        weights = original(similarity)
        recorded.append(weights.detach().clone())
        return weights

    exchange.attention = record
    with torch.no_grad():
        before = model.catalog_vectors().clone()
        weights_before = list(recorded)
        recorded.clear()
        titles = store.arrays["titles"]
        store.arrays["titles"] = np.where(titles != 0, (titles % 60) + 1, 0)
        model.clear_item_cache()
        after = model.catalog_vectors()
    assert len(weights_before) == len(recorded) > 0
    assert all(torch.equal(a, b) for a, b in zip(weights_before, recorded))
    assert not torch.allclose(before, after)


def test_existing_identities_are_unchanged_and_new_ones_record_the_exchange(context, monkeypatch):
    current = json.loads(json.dumps(experiment_identities(context, monkeypatch), default=str))
    assert current == SNAPSHOT["identity"]
    assert len(current["identities"]) == len(PRE_SCENE_ATTENTION_EXPERIMENTS) == 10

    rows = [{"label": "graph+GNN_ATTN", "source": "graph_qwen3vl_2b", "modules": ["GNN_ATTN"]},
            {"label": "all", "source": "graph_qwen3vl_2b", "modules": ["ASHA", "GNN_ATTN", "REAF"]}]
    contract = json.dumps(EXTENSION_MODELS, sort_keys=True)
    new = experiment_identities(context, monkeypatch, rows)["identities"]
    one, both = new["graph+GNN_ATTN"], new["all"]
    assert one["experiment_definition"]["modules"] == ["scene_attention"]
    assert one["experiment_definition"]["module_versions"] == {"scene_attention": "gnn-attn/v1"}
    assert one["experiment_arm"]["extensions"] == ["scene_attention"]
    assert one["experiment_arm"]["role_fusion"] is False
    assert one["item_model"]["extensions"] == {"scene_attention": EXTENSION_MODELS["scene_attention"]}
    assert both["experiment_definition"]["modules"] == ["role_fusion", "scene_attention", "activity_hypergraph"]
    assert both["experiment_arm"]["extensions"] == ["scene_attention", "segments"]
    assert both["item_model"]["extensions"]["segments"]["applied_after"] == "scene_attention_exchange"
    # Only the identity copy records the new position.
    assert json.dumps(EXTENSION_MODELS, sort_keys=True) == contract
    assert EXTENSION_MODELS["segments"]["applied_after"] == "scene_mlp_without_scene_exchange"


def test_config_accepts_exchange_combinations_and_rejects_the_rest():
    def parse(modules, source="graph_qwen3vl_2b", **extra):
        value = {"version": 1, "experiments": [{"label": "trial", "source": source, "modules": modules, **extra}]}
        return parse_experiments(value)

    for modules, expected in (
        (["GNN_ATTN"], ["scene_attention"]),
        (["REAF", "GNN_ATTN"], ["role_fusion", "scene_attention"]),
        (["GNN_ATTN", "ASHA"], ["scene_attention", "activity_hypergraph"]),
        (["ASHA", "GNN_ATTN", "REAF"], ["role_fusion", "scene_attention", "activity_hypergraph"]),
        (["scene_attention"], ["scene_attention"]),
    ):
        resolved = parse(modules)
        assert resolved["experiments"][0]["modules"] == expected
        arm = registry({CONFIG_KEY: resolved})["trial"]
        assert arm.extensions == tuple(e for e, m in (("scene_attention", "scene_attention"),
                                                       ("segments", "activity_hypergraph")) if m in expected)
    assert parse(["GNN_ATTN", "ASHA"], segments="random")["experiments"][0]["segments"]["variant"] == "random"
    for modules, source, extra in (
        (["GNN_ATTN"], "metadata", {}),
        (["GNN_ATTN"], "desc_qwen3vl_2b", {}),
        (["GNN_ATTN"], "graph_qwen3vl_2b", {"readout": "attn"}),
        (["GNN_ATTN", "scene_attention"], "graph_qwen3vl_2b", {}),
        (["GNN_ATTEN"], "graph_qwen3vl_2b", {}),
        (["GNN_ATTN"], "graph_qwen3vl_2b", {"segments": "random"}),
    ):
        with pytest.raises(ValueError):
            parse(modules, source, **extra)


def test_default_batches_and_existing_outputs_are_unchanged():
    torch.manual_seed(7)
    store = synthetic_store()
    packed = PackedGraphStore(store, feature_cache_mb=0)
    items = np.arange(len(store))
    for name, kwargs in {"prepared": {}, "raw": {"prepared": False}, "segments": {"scene_segments": True}}.items():
        batch = packed.batch(items, "cpu", **kwargs)
        assert {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in batch.items()} == SNAPSHOT["batches"][name]
    batch = packed.batch(items, "cpu", scene_groups=True)
    added = set(batch) - set(SNAPSHOT["batches"]["prepared"])
    expected = scene_groups(batch["offsets"])
    assert added == set(expected) and all(batch[k].tolist() == v.tolist() for k, v in expected.items())
    for name, extensions, role_fusion in (("graph", (), False), ("graph+REAF", (), True),
                                          ("graph+ASHA", ("segments",), False), ("HGVRec", ("segments",), True)):
        model = make_model(store, extensions, role_fusion=role_fusion)
        with torch.no_grad():
            if extensions:
                model.graph_encoder.segment_gates.copy_(torch.tensor([0.8, 0.6]))
            actual = model.catalog_vectors()
        torch.testing.assert_close(actual, torch.tensor(SNAPSHOT["outputs"][name]), atol=1e-6, rtol=1e-5)


def test_chunks_bound_scene_pairs_without_changing_outputs():
    store = synthetic_store()
    whole = make_model(store, EXCHANGE, items=1000, role_fusion=False)
    split = make_model(store, EXCHANGE, items=1000, role_fusion=False)
    split.execution = split.execution.model_copy(update={"chunk_scene_pairs": 30})
    sizes, original = [], split.packed.batch

    def record(items, device, **kwargs):
        sizes.append(int(np.square(split.packed.scene_counts[items]).sum()))
        assert kwargs.get("scene_groups") is True
        return original(items, device, **kwargs)

    split.packed.batch = record
    with torch.no_grad():
        expected, actual = whole.catalog_vectors(), split.catalog_vectors()
    assert len(sizes) > 1 and max(sizes) <= 30
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    # Without the exchange the bound is ignored, as before.
    plain = make_model(store, items=1000, role_fusion=False)
    plain.execution = plain.execution.model_copy(update={"chunk_scene_pairs": 1})
    with torch.no_grad():
        plain.catalog_vectors()
    assert plain.execution_counts["chunks"] == 1


@pytest.mark.parametrize("role_fusion", [False, True])
def test_closed_segment_gates_reduce_to_the_exchange_model(role_fusion):
    store = synthetic_store()
    reference = make_model(store, EXCHANGE, role_fusion=role_fusion)
    model = make_model(store, ("scene_attention", "segments"), role_fusion=role_fusion)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in reference.state_dict().items())
    with torch.no_grad():
        torch.testing.assert_close(model.catalog_vectors(), reference.catalog_vectors(), atol=0, rtol=0)
        model.graph_encoder.segment_gates.copy_(torch.tensor([0.8, 0.6]))
        model.clear_item_cache()
        assert not torch.allclose(model.catalog_vectors(), reference.catalog_vectors())
    report = model.execution_report()
    assert set(report["scene_attention_norms"]) == {"query", "key", "projection"}


def test_logs_show_config_module_names_while_identity_keeps_internal_ones():
    from rec_experiments import display_definition, experiment_definition

    resolved = parse_experiments({"version": 1, "experiments": [
        {"label": "all", "source": "graph_qwen3vl_2b", "modules": ["ASHA", "GNN_ATTN", "REAF"]}]})
    definition = experiment_definition(registry({CONFIG_KEY: resolved})["all"])
    shown = display_definition(definition)
    assert shown["modules"] == ["REAF", "GNN_ATTN", "ASHA"]
    assert shown["module_versions"] == {"REAF": "role-fusion/v1", "GNN_ATTN": "gnn-attn/v1", "ASHA": "segments-v1"}
    assert definition["modules"] == ["role_fusion", "scene_attention", "activity_hypergraph"]
