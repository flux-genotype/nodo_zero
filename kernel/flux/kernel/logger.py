import json
import uuid
import threading
from datetime import datetime, timezone
from typing import Optional
import atexit

class FluxLogger:
    def __init__(self, log_file: str = "flux.jsonl", tenant_id: str = "default"):
        self.log_file = log_file
        self.tenant_id = tenant_id
        self._lock = threading.Lock()
        self._file = open(log_file, "a", encoding="utf-8")
        atexit.register(self._close)

    def _close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def _generate_run_id(self):
        return str(uuid.uuid4())[:8]

    def log_event(self, event_type: str, data: dict = None, tenant_id: Optional[str] = None):
        if data is None:
            data = {}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "tenant_id": tenant_id or self.tenant_id,
            **data
        }
        with self._lock:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
