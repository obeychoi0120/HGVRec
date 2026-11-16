from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from visual_sampling import validate_sampling

CONFIG_PATH = Path("config.yaml")
DATASETS = ("microlens_100k", "m3l_10m")


class ConfigError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be an object: {path}")
    return value


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class RunContext:
    root: Path
    run_id: str
    config: dict[str, Any]
    run_root: Path
    artifact_root: Path
    dataset: str

    @classmethod
    def load(cls, run_id: str, *, dataset: str, root: Path | None = None) -> "RunContext":
        repo_root = (root or Path(__file__).resolve().parents[1]).resolve()
        selected = str(run_id).strip()
        if (
            not selected
            or selected in {".", "..", "resized_keyframes", "source_assets"}
            or Path(selected).name != selected
            or ("\\" in selected)
        ):
            raise ConfigError("run_id must be a single non-empty directory name")
        config = _load_yaml(repo_root / CONFIG_PATH)
        _validate_config(config)
        if dataset not in config["datasets"]:
            raise ConfigError(f"unsupported dataset: {dataset!r}; choose from {sorted(config['datasets'])}")
        artifact_root = _resolve(
            repo_root, config.get("artifacts_root", "artifacts"), "artifacts_root"
        )
        run_root = artifact_root / "runs" / selected / dataset
        return cls(repo_root, selected, config, run_root, artifact_root, dataset)

    def initialize(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.cohort_dir,
            self.run_root / "extraction",
            self.run_root / "validation",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def require_ready_cohort(self) -> dict[str, Any]:
        from preparation.cohort import load_cohort

        return load_cohort(self.cohort_dir, dataset=self.dataset)

    @property
    def data(self) -> dict[str, Any]:
        return self.config["datasets"][self.dataset]["data"]

    @property
    def source_counts(self) -> dict[str, int]:
        """Optional expected counts; used only for progress totals."""
        return self.config["datasets"][self.dataset].get("source_counts", {})

    @property
    def preparation_dir(self) -> Path:
        return self.artifact_root / "preparation" / self.dataset

    @property
    def shared_cache_dir(self) -> Path:
        return self.artifact_root / "shared_cache" / self.dataset

    @property
    def cohort_dir(self) -> Path:
        return self.preparation_dir / "cohort" / "leave_one_out"

    @property
    def evidence_dir(self) -> Path:
        return self.preparation_dir

    @property
    def keyframes_dir(self) -> Path:
        width, height = self.config["extraction"]["visual_evidence"]["image_resolution"]
        return self.evidence_dir / "resized_keyframes" / f"{width}_{height}"

    @property
    def source_assets_dir(self) -> Path:
        return self.evidence_dir / "source_assets"

    def extraction_dir(self, representation: str, model: str, phase: str = "scenes") -> Path:
        from arm_registry import generated_arm

        return self.scene_arm_dir(generated_arm(representation, model).name)

    def scene_arm_dir(self, arm):
        return self.run_root / "extraction" / arm

    def graph_scene_dir(self, source: str) -> Path:
        return self.extraction_dir("graph", source, "scenes")

    def description_scene_dir(self, source: str) -> Path:
        return self.extraction_dir("description", source, "scenes")

    def prompt_path(self, schema: str | Path) -> Path:
        path = _resolve(self.root, str(schema), "--schema")
        if not path.is_file():
            raise ValueError(f"--schema prompt not found: {path}")
        return path

    @property
    def validation_dir(self) -> Path:
        return self.run_root / "validation"

    @property
    def representations_dir(self) -> Path:
        return self.validation_dir / "representations"

    @property
    def recommendations_dir(self) -> Path:
        return self.validation_dir / "recommendations"

    @property
    def diagnosis_dir(self) -> Path:
        return self.validation_dir / "diagnosis"

    def diagnosis_path(self, target) -> Path:
        from arm_registry import target_set
        from rec_experiments import load_experiments

        load_experiments(self)
        return self.diagnosis_dir / target_set(self.config, target) / "diagnosis.json"

    def path(self, section: str, key: str) -> Path:
        values = self.data if section == "data" else _require_mapping(self.config, section)
        return _resolve(self.root, values.get(key), f"{section}.{key}")


def require_generation_config(config: dict[str, Any], representation: str, model: str) -> None:
    """Fail before a generation step starts if its model was omitted from the config."""
    missing = [
        name for name, present in (
            (f"models.{model}", model in config["models"]),
            (f"extraction.{representation}.{model}", model in config["extraction"].get(representation, {})),
        ) if not present
    ]
    if missing:
        raise ConfigError(f"{model} {representation} generation requires {', '.join(missing)}")


def _validate_config(value: dict[str, Any]) -> None:
    """Only check what every step needs to start; everything else is read where it is used."""
    for key in ("datasets", "models", "extraction", "validation"):
        _require_mapping(value, key)
    visual_evidence = _require_mapping(value["extraction"], "visual_evidence")
    try:
        validate_sampling(visual_evidence["scene_duration"], visual_evidence["num_keyframes"])
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"extraction.visual_evidence: {exc}") from exc
