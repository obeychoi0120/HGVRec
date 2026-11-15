import pytest

from artifact_io import read_jsonl
from extraction.graph_warnings import PARSE_FAILED, REPEAT_OCCURRED, TOKEN_LIMIT
from extraction.qwen_runtime import QwenRuntime
from extraction.scene_executor import graph_scene_result, run_qwen_scenes
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.semantic_graph import parse_or_repair_graph
from extraction.steps import prepare_scene_batch
from extraction.step_support import minimal_graph_records, scene_generation_rows
from helpers import (
    GRAPH_TEXT,
    NAMED_GRAPH_TEXT,
    FakeGenerator,
    failure_row,
    graph,
    quiet_progress,
    ready_cohort,
    scene_row,
)


def test_named_graph_parsing_and_repairs():
    # Native named output, and invalid output preserved as text.
    parsed = parse_or_repair_graph(NAMED_GRAPH_TEXT)
    assert parsed.error is None and not parsed.warning
    expected = graph()
    for entity, name in zip(
        expected["entities"], ["person 1", "tomato 2", "knife", "kitchen"], strict=True
    ):
        entity["name"] = name
    for index, entity in enumerate(expected["entities"], 1):
        entity["id"] = f"e{index}"
    expected["actions"][0].update(actor="e1", target="e2", tool="e3", place="e4")
    assert parsed.graph == expected

    invalid = GRAPH_TEXT.split("[Actions]")[0]
    record, failure = graph_scene_result({"scene_idx": 7, "keyframes": [2.5]}, invalid)
    assert failure is None
    assert record["graph"] == invalid
    assert record["parse_mode"] == "text"
    assert record["semantic_warnings"] == [PARSE_FAILED]
    assert record["scene_idx"] == 7

    # Roles missing from [Entities] are kept as new entities.
    text = (
        "[Context]\nmedium: live_action\nformat: narrative\ntopics: hospital visit\n"
        "[Entities]\nman; wearing coat\n"
        "[Actions]\nman - removing - sunglasses; None; hospital room\n"
        "man - holding - sunglasses; none; unknown"
    )
    parsed = parse_or_repair_graph(text)
    assert parsed.error is None and not parsed.warning and parsed.parse_mode == "repaired"
    assert parsed.graph["entities"] == [
        {"id": "e1", "name": "man", "attributes": ["wearing coat"]},
        {"id": "e2", "name": "sunglasses", "attributes": []},
        {"id": "e3", "name": "hospital room", "attributes": []},
    ]
    assert parsed.graph["actions"] == [
        {"actor": "e1", "action": "removing", "target": "e2", "tool": None, "place": "e3"},
        {"actor": "e1", "action": "holding", "target": "e2", "tool": None, "place": "unknown"},
    ]
    # ID-form output is unchanged: an undeclared ID has no name to keep.
    legacy = parse_or_repair_graph(
        "[Context]\nmedium: live_action\nformat: narrative\ntopics: hospital visit\n"
        "[Entities]\np1: man\n[Actions]\np1 - removing - t9; none; unknown"
    )
    assert legacy.graph["entities"] == [{"id": "p1", "name": "man", "attributes": []}]
    assert legacy.graph["actions"][0]["target"] == "unknown"

    # Actions joined on one line are regrouped.
    head = "[Context]\nmedium: animation\nformat: narrative\ntopics: dorm\n[Entities]\n"
    names = "woman1; white blouse\nman1\ndoor\nknife\nroom\n[Actions]\n"

    def actions(text):
        parsed = parse_or_repair_graph(head + names + text)
        assert parsed.error is None and parsed.parse_mode == "repaired", parsed.error
        return [tuple(a.values()) for a in parsed.graph["actions"]]

    # e1 woman1, e2 man1, e3 door, e4 knife, e5 room
    assert actions("woman1 - pushing - door; man1 - looking - woman1; man1 - cutting - door") == [
        ("e1", "pushing", "e3", None, "unknown"),
        ("e2", "looking", "e1", None, "unknown"),
        ("e2", "cutting", "e3", None, "unknown"),
    ]
    assert actions("man1 - cutting - door; knife; room; woman1 - leaving\nman1 - waiting - none; knife") == [
        ("e2", "cutting", "e3", "e4", "e5"),
        ("e1", "leaving", "unknown", None, "unknown"),
        ("e2", "waiting", None, "e4", "unknown"),
    ]
    # Pieces that cannot be told apart fail: a slot list, a leading orphan, a fourth part.
    for loose in ("man1 - cutting - door; knife; room; door", "knife; man1 - cutting - door",
                  "man1 - cutting - door - room"):
        assert parse_or_repair_graph(head + names + loose).warning == (PARSE_FAILED,), loose

    # ID-form output takes the same repair.
    legacy = parse_or_repair_graph(head + "p1: woman\nf1: door\n[Actions]\np1 - pushing - f1; p1 - leaving - f1")
    assert legacy.parse_mode == "repaired" and [tuple(a.values()) for a in legacy.graph["actions"]] == [
        ("p1", "pushing", "f1", None, "unknown"), ("p1", "leaving", "f1", None, "unknown")]

    # Labelled optional tool and place.
    head = ("[Context]\nmedium: animation\nformat: narrative\ntopics: dorm\n"
            "[Entities]\nwoman\ndoor\nknife\nroom\n[Actions]\n")

    def labelled(text, mode="native"):
        parsed = parse_or_repair_graph(head + text)
        assert parsed.error is None and parsed.parse_mode == mode, parsed.error
        return [tuple(a.values()) for a in parsed.graph["actions"]]

    # e1 woman, e2 door, e3 knife, e4 room. An omitted tool is none, an omitted place unknown.
    assert labelled("woman - cutting - door; tool: knife; place: room\nwoman - pushing - door; place: room\n"
                   "woman - prying - door; tool: knife\nwoman - leaving - none") == [
        ("e1", "cutting", "e2", "e3", "e4"), ("e1", "pushing", "e2", None, "e4"),
        ("e1", "prying", "e2", "e3", "unknown"), ("e1", "leaving", None, None, "unknown")]
    assert labelled("woman - cutting - door; place: room; tool: knife\nwoman - pushing - door; tool: none") == [
        ("e1", "cutting", "e2", "e3", "e4"), ("e1", "pushing", "e2", None, "unknown")]
    # Positional slots still parse, and loose labels are repaired.
    assert labelled("woman - cutting - door; knife; room") == [("e1", "cutting", "e2", "e3", "e4")]
    assert labelled("woman - cutting - door; Tool : knife; room\nwoman - pushing - door; place:",
                   "repaired") == [("e1", "cutting", "e2", "e3", "e4"), ("e1", "pushing", "e2", None, "unknown")]
    assert labelled("woman - cutting - door; tool: knife; woman - pushing - door; place: room", "repaired") == [
        ("e1", "cutting", "e2", "e3", "unknown"), ("e1", "pushing", "e2", None, "e4")]
    # A repeated label or a slot beyond tool and place fails.
    for text in ("woman - cutting - door; tool: knife; tool: room",
                 "woman - cutting - door; tool: knife; place: room; door", "place: room"):
        assert parse_or_repair_graph(head + text).warning == (PARSE_FAILED,), text
    # ID-form output also takes labelled slots.
    legacy = head.replace("woman\ndoor\nknife\nroom", "p1: woman\nf1: door\nt1: knife")
    parsed = parse_or_repair_graph(legacy + "p1 - cutting - f1; tool: t1")
    assert parsed.parse_mode == "repaired" and parsed.graph["actions"][0]["tool"] == "t1"

    # A missing [Context] header and a stray none are repaired.
    body = "medium: animation\nformat: narrative\ntopics: dorm\n[Entities]\n{entities}\n[Actions]\n{actions}"
    parsed = parse_or_repair_graph(body.format(entities="woman; white blouse\ndoor",
                                               actions="none\nwoman - pushing - door; none; unknown"))
    assert parsed.error is None and parsed.parse_mode == "repaired", parsed.error
    assert parsed.graph["medium"] == "animation"
    assert [tuple(a.values()) for a in parsed.graph["actions"]] == [("e1", "pushing", "e2", None, "unknown")]
    both = parse_or_repair_graph("medium: animation\nformat: narrative\ntopics: dorm\n"
                                 "woman; white blouse\ndoor\n[Actions]\nwoman - pushing - door; none; unknown")
    assert both.parse_mode == "repaired" and [e["name"] for e in both.graph["entities"]] == ["woman", "door"]
    # ID-form output takes the same repairs.
    legacy = "p1: woman; white blouse\no1: door"
    parsed = parse_or_repair_graph(body.format(entities=legacy, actions="none"))
    assert parsed.parse_mode == "repaired" and not parsed.warning and parsed.graph["medium"] == "animation"
    parsed = parse_or_repair_graph("[Context]\n" + body.format(
        entities=legacy, actions="none\np1 - pushing - o1; none; unknown"))
    assert [tuple(a.values()) for a in parsed.graph["actions"]] == [("p1", "pushing", "o1", None, "unknown")]
    # A missing [Entities] header after [Context] is inserted where the context lines end.
    parsed = parse_or_repair_graph("[Context]\nmedium: animation\nformat: narrative\ntopics: dorm\n"
                                   "woman\n[Actions]\nnone")
    assert not parsed.warning and [e["name"] for e in parsed.graph["entities"]] == ["woman"]


HEAD = "[Context]\nmedium: live_action\nformat: narrative\ntopics: kitchen\n[Entities]\n"

# Output whose sections or slots cannot be separated by the one repair pass fails.
FAILED_CASES = [
    "A man cooks pasta in a kitchen.",
    # Slot list after the target: tool and place cannot be told apart.
    HEAD + "woman\nman\n[Actions]\nwoman - speaking - man; man; door; car",
    # No [Entities] or [Actions] headers at all.
    "medium: animation\nformat: narrative\ntopics: duel\nman; brown hair\nsky\nman - flying - none; none; sky",
    # A fourth action part, and an action without spaces around its text.
    HEAD + "man\nfish\npit\n[Actions]\nman - placing - fish - into pit; none; pit",
    HEAD + "man\nfish\n[Actions]\nman -捞取- fish; none; unknown",
    # Reserved entity names and labels other than tool and place.
    HEAD + "unknown; dark room\n[Actions]\nnone",
    HEAD + "man\nmap\n[Actions]\nman - pulling - map; target: chest",
    # One-line sections.
    "Context: animation; narrative; mountains\nEntities: mountain; cloud\nActions: none",
    # Missing context field, missing or empty sections.
    HEAD.replace("topics: kitchen\n", "") + "man\n[Actions]\nnone",
    HEAD + "man\n[Actions]\nplace: kitchen",
    HEAD + "man",
    "medium: animation\nformat: narrative\ntopics: wolves\nwolf1; gray fur\nforest",
]


def test_failed_parsing_and_prompt_examples():
    from pathlib import Path

    for text in FAILED_CASES:
        parsed = parse_or_repair_graph(text)
        assert parsed.graph is None and parsed.warning == (PARSE_FAILED,), text
    # A second graph after the first is content after Actions.
    assert parse_or_repair_graph(
        HEAD + "cat\n[Actions]\nnone\n[Context]\nmedium: animation").warning == (PARSE_FAILED,)
    # Every example in the ID-form (v8, v8_2) and name-form (v9, v10) prompts parses natively.
    for name in ("v8", "v8_2", "v9_core", "v9_extended", "v10_core", "v10_extended"):
        text = (Path(__file__).parents[1] / "prompts" / f"scene_graph_{name}.md").read_text(encoding="utf-8")
        blocks = [block for block in text.split("\n\n") if "\n[Context]\nmedium: " in "\n" + block]
        examples = [block[block.index("[Context]"):] for block in blocks]
        examples = [example for example in examples if "medium: value" not in example]
        assert len(examples) >= 2, name
        for example in examples:
            parsed = parse_or_repair_graph(example)
            assert parsed.error is None and parsed.parse_mode == "native", (name, example, parsed.error)
    # v8_2 role IDs are kept as declared.
    parsed = parse_or_repair_graph(
        HEAD + "actor_1: woman; apron\ntarget_1: salmon fillet\n[Actions]\n"
        "actor_1 - slicing thinly - target_1; none; unknown")
    assert [e["id"] for e in parsed.graph["entities"]] == ["actor_1", "target_1"]
    assert parsed.graph["actions"][0]["actor"] == "actor_1"


def test_scene_storage_resume_and_progress(context, tmp_path):
    from extraction.progress import InferenceProgress
    from tqdm import tqdm
    import io

    # Compact storage round trip.
    cid = "video"
    path = tmp_path / f"{cid}.jsonl"
    records = [
        failure_row(cid, 2),
        scene_row(cid),
        {"content_id": cid, "scene_idx": 1, "tokens": 8, "description": "A person cuts."},
    ]
    write_scene_records(path, records)
    assert read_jsonl(path) == sorted(records, key=lambda r: r["scene_idx"])
    restored = read_scene_records(path, include_failures=True)
    assert [r["scene_idx"] for r in restored] == [0, 1, 2]
    assert restored[0]["graph"] == graph() and restored[0]["tokens"] == 19
    assert restored[1]["description"] == "A person cuts."
    assert restored[2] == failure_row(cid, 2)
    assert len(read_scene_records(path)) == 2
    # Replacing an attempt must leave exactly one row for that scene.
    write_scene_records(path, [records[1], {**records[2], "description": "Updated."}, records[0]])
    assert read_jsonl(path)[1]["description"] == "Updated."
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate scene"):
        write_scene_records(path, [records[1], records[1]])
    assert path.read_bytes() == before

    # Resume selects pending scenes only.
    directory = context.scene_arm_dir("graph_qwen3vl_2b")
    path = directory / f"{cid}.jsonl"
    write_scene_records(
        path, [scene_row(cid), failure_row(cid, 1), scene_row(cid, 2, warnings=[PARSE_FAILED])]
    )
    visual = {"content_id": cid}

    def prepare(visual):
        return [
            {
                "scene_idx": i,
                "keyframes": [i + 0.5],
                "scene_start_seconds": i,
                "scene_end_seconds": i + 1,
            }
            for i in range(4)
        ]

    before = path.read_bytes()
    existing, pending = prepare_scene_batch(
        [visual], directory, prepare, minimal_graph_records, force=False
    )
    assert len(existing[cid]) == 3
    assert [r["scene_idx"] for r in pending[0][1]] == [1, 2, 3]
    assert path.read_bytes() == before
    _, forced = prepare_scene_batch([visual], directory, prepare, minimal_graph_records, force=True)
    assert [r["scene_idx"] for r in forced[0][1]] == [0, 1, 2, 3]

    # Progress counts each scene in exactly one outcome.
    stream = io.StringIO()
    with InferenceProgress(total=5, desc="x", unit="scene",
                           progress_factory=lambda **kw: tqdm(file=stream, **kw)) as progress:
        for task_id, result in zip("abcde", ("success", "repeat_occurred", "parse_failed", "token_limit", "error")):
            progress.complete(task_id=task_id, outcome=result)
        progress.complete(task_id="e", outcome="success")  # a later pass replaces the outcome
        counts = dict(progress.counts)
        with pytest.raises(ValueError, match="unknown scene outcome"):
            progress.complete(outcome="warning")
    assert counts == {"success": 2, "repeat_occurred": 1, "parse_failed": 1, "token_limit": 1, "error": 0}
    assert "success=2 repeat_occurred=1 parse_failed=1 token_limit=1 error=0" in stream.getvalue()


@pytest.mark.torch
def test_qwen_scene_runs_warn_on_token_limit_and_retry_with_penalty(context, monkeypatch):
    from preparation.input_data import prepare_input_data
    from extraction.step_support import visual_rows
    from helpers import install_doubles

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    visual = visual_rows(context)[0]
    rows = scene_generation_rows(visual, prompt="observe", max_new_tokens=384)[:1]
    factory = FakeGenerator(finish_reason="length")
    runtime = QwenRuntime()
    directory = context.scene_arm_dir("graph_qwen3vl_2b")
    run_qwen_scenes(
        [(visual, rows)],
        scene_dir=directory,
        model_path=context.path("models", "qwen3vl_2b"),
        generator_factory=factory,
        names={},
        progress=quiet_progress(),
        arm="graph",
        existing_records={},
        runtime=runtime,
    )
    saved = read_jsonl(directory / f"{visual['content_id']}.jsonl")
    assert len(saved) == 1
    # A cut output is a TOKEN_LIMIT warning that keeps the unparsed text.
    assert saved[0]["warning"] == [TOKEN_LIMIT] and saved[0]["scene_graph"] == GRAPH_TEXT
    assert saved[0]["tokens"] == 19 and "error" not in saved[0]

    # Penalty passes retry warned scenes: prose first (PARSE_FAILED), a graph at the higher penalty.
    factory = FakeGenerator(lambda task: GRAPH_TEXT if task.repetition_penalty > 1 else "A man cooks.")
    directory = context.scene_arm_dir("graph_qwen3vl_8b")
    run_qwen_scenes(
        [(visual, rows)], scene_dir=directory, model_path=context.path("models", "qwen3vl_2b"),
        generator_factory=factory, names={}, progress=quiet_progress(), arm="graph",
        existing_records={}, runtime=QwenRuntime(), penalties=(1.0, 1.1),
    )
    assert [task.repetition_penalty for task in factory.tasks] == [1.0, 1.1]
    assert read_jsonl(directory / f"{visual['content_id']}.jsonl")[0]["warning"] == []

    # REPEAT_OCCURRED is retried too: a later success is stored as a plain success, and a
    # retry that is no longer usable keeps the earlier repeated graph.
    def repeat_run(name, text):
        factory = FakeGenerator(text, repeat_stopped=lambda task: task.repetition_penalty == 1.0)
        directory = context.scene_arm_dir(name)
        run_qwen_scenes(
            [(visual, rows)], scene_dir=directory, model_path=context.path("models", "qwen3vl_2b"),
            generator_factory=factory, names={}, progress=quiet_progress(), arm="graph",
            existing_records={}, runtime=QwenRuntime(), penalties=(1.0, 1.1, 1.2),
        )
        return factory, read_jsonl(directory / f"{visual['content_id']}.jsonl")[0]

    factory, saved = repeat_run("graph_qwen3vl_2b", GRAPH_TEXT)
    assert [task.repetition_penalty for task in factory.tasks] == [1.0, 1.1]
    assert saved["warning"] == [] and isinstance(saved["scene_graph"], dict)
    factory, saved = repeat_run(
        "graph_gemini", lambda task: GRAPH_TEXT if task.repetition_penalty == 1.0 else "A man cooks.")
    assert [task.repetition_penalty for task in factory.tasks] == [1.0, 1.1, 1.2]
    assert saved["warning"] == [REPEAT_OCCURRED] and isinstance(saved["scene_graph"], dict)


def test_scene_sources_extract_reuse_and_retry(context, monkeypatch):
    from helpers import install_doubles
    from preparation.input_data import prepare_input_data

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    for model in ("qwen3vl_2b", "gemini"):
        for representation in ("graph", "description"):
            _extract_reuse_and_retry(context, monkeypatch, model, representation)


def _extract_reuse_and_retry(context, monkeypatch, model, representation):
    from extraction.backends.gemini_workers import GeminiGenerationOutcome
    from extraction.steps import extract_description_scenes, extract_graph_scenes

    graph_source = representation == "graph"
    generator = FakeGenerator(GRAPH_TEXT if graph_source else "A person cuts a tomato.")

    class GeminiPool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, complete, **kwargs):
            for task in tasks:
                generator.tasks.append(task)
                complete(GeminiGenerationOutcome(task.task_id, generator.text))

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    monkeypatch.setattr("extraction.steps.GeminiWorkerPool", GeminiPool)
    extract = extract_graph_scenes if graph_source else extract_description_scenes
    options = dict(
        model=model,
        arm=f"{'graph' if graph_source else 'desc'}_{model}",
        schema="prompts/scene_graph_v9_core.md" if graph_source else "prompts/scene_description_v3.md",
    )
    assert extract(context, **options)["failure_count"] == 0
    assert generator.tasks
    for task in generator.tasks:
        assert task.max_new_tokens == context.config["extraction"][representation][model]["scene_max_new_tokens"]
        assert "TITLE_SECRET" not in task.prompt
        if model != "gemini":
            assert task.presence_penalty == (0.0 if graph_source else 1.5)
            assert task.repetition_penalty == (1.05 if graph_source else 1.0)
    generator.tasks.clear()
    assert extract(context, **options)["failure_count"] == 0
    assert not generator.tasks

    path = next(context.scene_arm_dir(options["arm"]).glob("*.jsonl"))
    rows = read_jsonl(path)
    rows[0] = failure_row(path.stem, rows[0]["scene_idx"])
    write_scene_records(path, rows)
    assert extract(context, **options)["failure_count"] == 0
    assert len(generator.tasks) == 1
    assert all("error" not in row for row in read_scene_records(path, include_failures=True))


LOOPED_GRAPH_TEXT = (
    "[Context]\nmedium: animation\nformat: narrative\ntopics: forest scene\n"
    "[Entities]\npanda1; white robe\npanda2; orange robe\njar\n"
    "[Actions]\n"
    + "panda1 - greeting - panda2; none; unknown\npanda2 - lifting - jar; none; unknown\n" * 9
    + "panda1 - greet"
)


def test_truncated_scenes_are_token_limit_warnings_and_regenerate(context, monkeypatch):
    from artifact_io import atomic_write_jsonl, read_json
    from extraction.steps import extract_description_scenes, extract_graph_scenes
    from helpers import install_doubles
    from preparation.input_data import prepare_input_data

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    generator = FakeGenerator(LOOPED_GRAPH_TEXT, finish_reason="length")
    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    options = dict(model="qwen3vl_2b", arm="graph_qwen3vl_2b", schema="prompts/scene_graph_v9_core.md")
    assert extract_graph_scenes(context, **options)["failure_count"] == 0
    total = len(generator.tasks)
    directory = context.scene_arm_dir(options["arm"])
    assert read_json(directory / "generation_stats.json") == {
        "total": total, "success": 0, "repeat_occurred": 0, "parse_failed": 0, "token_limit": total, "error": 0}
    path = next(directory.glob("*.jsonl"))
    assert all(row["warning"] == [TOKEN_LIMIT] and row["scene_graph"] == LOOPED_GRAPH_TEXT
               for row in read_jsonl(path))

    # Warned scenes are generated again, also when --arm is omitted.
    generator.text, generator.finish_reason = GRAPH_TEXT, "stop"
    generator.tasks.clear()
    assert extract_graph_scenes(context, model="qwen3vl_2b", schema=options["schema"])["failure_count"] == 0
    assert len(generator.tasks) == total
    assert read_json(directory / "generation_stats.json")["success"] == total
    with pytest.raises(ValueError, match="invalid graph generation arm"):
        extract_graph_scenes(context, model="qwen3vl_2b", arm="desc_qwen3vl_2b", schema=options["schema"])

    # Stored rows follow the current rules first: a legacy truncation error becomes TOKEN_LIMIT
    # and is generated again; PARSE_FAILED text the current parser reads is kept without generation.
    rows = read_jsonl(path)
    rows[0] = {"content_id": path.stem, "scene_idx": rows[0]["scene_idx"], "tokens": 384,
               "error": "graph: output truncated at token limit", "raw_output": LOOPED_GRAPH_TEXT}
    headless = "medium: animation\nformat: narrative\ntopics: duel\nman\nsky\n[Actions]\nman - flying - none; none; sky"
    rows[-1] = {**rows[-1], "warning": ["MISSING_REQUIRED"], "scene_graph": headless}
    atomic_write_jsonl(path, rows, durable=True, sort_keys=False)
    generator.tasks.clear()
    assert extract_graph_scenes(context, **options)["failure_count"] == 0
    assert len(generator.tasks) == 1
    saved = read_jsonl(path)
    assert saved[0]["warning"] == [] and isinstance(saved[0]["scene_graph"], dict)
    if len(rows) > 1:
        assert saved[-1]["warning"] == [] and saved[-1]["scene_graph"]["entities"][0]["name"] == "man"

    # An output stopped before a repeated line is used and counted, never generated again;
    # one that does not parse stays PARSE_FAILED.
    from extraction.graph_warnings import blocking_tags, needs_generation
    from extraction.scene_executor import graph_scene_result
    from extraction.scene_storage import read_scene_records

    generator.text, generator.finish_reason, generator.repeat_stopped = GRAPH_TEXT, "stop", True
    generator.tasks.clear()
    assert extract_graph_scenes(context, force=True, **options)["failure_count"] == 0
    assert read_json(directory / "generation_stats.json")["repeat_occurred"] == len(generator.tasks)
    assert all(row["warning"] == [REPEAT_OCCURRED] and isinstance(row["scene_graph"], dict)
               for row in read_jsonl(path))
    assert not any(needs_generation(row) for row in read_scene_records(path))
    generator.tasks.clear()
    extract_graph_scenes(context, **options)
    assert not generator.tasks
    assert blocking_tags([REPEAT_OCCURRED]) == [] and blocking_tags([PARSE_FAILED]) == [PARSE_FAILED]
    prose, _ = graph_scene_result({"scene_idx": 0, "keyframes": []}, "A man cooks.", repeated=True)
    assert prose["semantic_warnings"] == [PARSE_FAILED]
    generator.repeat_stopped = False

    # Descriptions use only TOKEN_LIMIT, and regenerate it the same way.
    generator.text, generator.finish_reason = "A person cuts a tomato and", "length"
    generator.tasks.clear()
    desc = dict(model="qwen3vl_2b", arm="desc_qwen3vl_2b", schema="prompts/scene_description_v3.md")
    assert extract_description_scenes(context, **desc)["failure_count"] == 0
    desc_dir = context.scene_arm_dir(desc["arm"])
    assert read_json(desc_dir / "generation_stats.json") == {
        "total": len(generator.tasks), "success": 0, "token_limit": len(generator.tasks), "error": 0}
    desc_path = next(desc_dir.glob("*.jsonl"))
    assert all(row["warning"] == [TOKEN_LIMIT] for row in read_jsonl(desc_path))
    generator.finish_reason = "stop"
    generator.tasks.clear()
    extract_description_scenes(context, **desc)
    assert generator.tasks and all("warning" not in row for row in read_jsonl(desc_path))
