"""Cross-run reuse of finished bundles, keyed by a content hash and published by rename."""

import shutil
import tempfile
from pathlib import Path

from artifact_io import write_json


class SharedCache:
    def __init__(self, context, kind, key):
        self.root = context.shared_cache_dir / kind
        self.key = key
        self.path = self.root / key

    def exists(self):
        return (self.path / "cache.json").is_file()

    def restore(self, destination):
        """Copy a published bundle into destination; False when the key was never published."""
        if not self.exists():
            return False
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for path in self.path.iterdir():
            if path.name != "cache.json":
                shutil.copy2(path, destination / path.name)
        return True

    def publish(self, source, names, *, origin):
        if self.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.root, prefix=f".{self.key}.") as temp:
            staging = Path(temp) / "bundle"
            staging.mkdir()
            for name in names:
                shutil.copy2(Path(source) / name, staging / name)
            write_json(staging / "cache.json", {"key": self.key, "origin": origin})
            if self.path.exists():
                shutil.rmtree(self.path)
            staging.replace(self.path)
