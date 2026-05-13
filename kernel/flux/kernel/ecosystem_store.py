import os
import json
import threading
from datetime import datetime, timezone
from typing import Optional
from flux.core.entities import Ecosystem
from flux.parser.flux_serializer import serialize_ecosystem, deserialize_ecosystem_file

class EcosystemStore:
    def __init__(self, base_path: str = "./ecosystems"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.index_file = os.path.join(base_path, "index.json")
        self._lock = threading.Lock()
        self._ensure_index()

    def _ensure_index(self):
        with self._lock:
            if not os.path.exists(self.index_file):
                with open(self.index_file, 'w') as f:
                    json.dump({"current_version": None, "versions": []}, f)

    def save_ecosystem(self, eco: Ecosystem, version: str = None):
        version = version or f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        file_path = os.path.join(self.base_path, f"{eco.name}_{version}.flux")
        code = serialize_ecosystem(eco)
        with open(file_path, 'w') as f:
            f.write(code)
        with self._lock:
            with open(self.index_file, 'r+') as f:
                index = json.load(f)
                index["versions"].append({"version": version, "file": file_path, "name": eco.name})
                f.seek(0)
                json.dump(index, f, indent=2)
                f.truncate()

    def set_active_version(self, version: str):
        with self._lock:
            with open(self.index_file, 'r+') as f:
                index = json.load(f)
                if any(v["version"] == version for v in index["versions"]):
                    index["current_version"] = version
                    f.seek(0)
                    json.dump(index, f, indent=2)
                    f.truncate()
                else:
                    raise ValueError(f"Version {version} not found.")

    def get_active_version(self) -> Optional[str]:
        with self._lock:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
            return index.get("current_version")

    def load_active_ecosystem(self) -> Optional[Ecosystem]:
        version = self.get_active_version()
        if not version:
            return None
        with self._lock:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
        for v in index["versions"]:
            if v["version"] == version:
                return deserialize_ecosystem_file(v["file"])
        return None
