"""Runtime MongoDB configuration.

Host-specific values live in config.local.json. Credentials are not stored in
the repository.
"""

import json
from typing import Any

from paths import LOCAL_CONFIG


def _read_local_config() -> dict[str, Any]:
    if not LOCAL_CONFIG.exists():
        return {}
    with LOCAL_CONFIG.open() as config_file:
        data = json.load(config_file)
    db_config = data.get("db", data)
    if not isinstance(db_config, dict):
        raise ValueError("config.local.json must contain a JSON object or a top-level 'db' object")
    return db_config


_local_config = _read_local_config()

db_config = {
    "connection_string": str(_local_config.get("connection_string", "")),
    "database": str(_local_config.get("database", "knn")),
    "collection": str(_local_config.get("collection", "knn")),
    "scheduler_collection": str(_local_config.get("scheduler_collection", "knn_scheduler")),
}
