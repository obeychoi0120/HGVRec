"""Qwen3-VL-4B sources run through every pipeline step under the ``qwen3vl_4b`` key."""

import pytest

from artifact_io import read_json
from helpers import FakeGenerator, install_doubles, ready_cohort

EXPERIMENTS = [
    {"label": "base", "source": "metadata", "modules": []},
    {"label": "description-4b", "source": "desc_qwen3vl_4b", "modules": []},
    {"label": "graph-4b", "source": "graph_qwen3vl_4b", "modules": ["REAF", "ASHA"]},
]


@pytest.mark.torch
def test_qwen3vl_4b_sources_run_extraction_to_diagnosis(context, monkeypatch, capsys):
    from extraction import cli as extraction_cli
    from pipeline_runtime import RunContext
    from preparation.input_data import prepare_input_data
    from rec_experiments import CONFIG_KEY, parse_experiments
    from validation.cli import main
    from validation.representation_provenance import recommendation_identity

    context.config[CONFIG_KEY] = parse_experiments({"version": 1, "experiments": EXPERIMENTS})
    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset, **kw: context)
    base = ["--run-id", context.run_id, "--dataset", context.dataset, "--model", "qwen3vl_4b"]

    graph = FakeGenerator()
    monkeypatch.setattr("extraction.steps.qwen_generator", graph)
    assert extraction_cli.main(["extract-graph-scenes", *base, "--schema", "prompts/scene_graph_v9_core.md"]) == 0
    description = FakeGenerator("A person cuts a tomato with a knife in a kitchen.")
    monkeypatch.setattr("extraction.steps.qwen_generator", description)
    assert extraction_cli.main([
        "extract-description-scenes", *base, "--schema", "prompts/scene_description_v3.md",
        "--arm", "desc_qwen3vl_4b",
    ]) == 0
    assert len(graph.tasks) == len(description.tasks) == 48
    for source in ("graph_qwen3vl_4b", "desc_qwen3vl_4b"):
        identity = read_json(context.scene_arm_dir(source) / "generation_identity.json")["identity"]
        assert identity["scene_arm"] == source
        assert identity["settings"]["generation"]["model"] == "qwen3vl_4b"
    assert not context.scene_arm_dir("graph_qwen3vl_2b").exists()

    labels = [row["label"] for row in EXPERIMENTS]
    def command(step):
        return main([step, "--run-id", context.run_id, "--dataset", context.dataset])
    assert command("embed-representations") == 0
    assert {p.name for p in context.representations_dir.iterdir()} == {
        "metadata", "desc_qwen3vl_4b", "graph_qwen3vl_4b",
    }
    assert command("prepare-module") == 0
    identity = recommendation_identity(context, "graph-4b")
    assert identity["experiment_definition"]["source"] == "graph_qwen3vl_4b"
    assert command("run-recommendation") == 0
    assert len(list(context.recommendations_dir.rglob("complete.json"))) == 3 * len(labels)
    capsys.readouterr()
    assert command("run-diagnosis") == 0
    assert read_json(context.diagnosis_path(labels))["targets"] == labels
