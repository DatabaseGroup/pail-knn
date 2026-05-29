"""MongoDB helpers for benchmark results and scheduler state."""

import json
from typing import List, Mapping, Union

# recursive type for json, see: https://github.com/python/typing/issues/182
JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]


def read_run(run_filename: str) -> JSON:
    with open(run_filename) as run_file:
        return json.loads(run_file.read())


def read_config(config_filename: str) -> JSON:
    with open(config_filename) as config_file:
        return json.loads(config_file.read())


def _require_pymongo():
    try:
        import pymongo as mng
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymongo is required for benchmark result and scheduler storage") from exc
    return mng


def _require_connection_string(db_config: JSON) -> str:
    connection_string = str(db_config.get('connection_string', ''))
    if not connection_string:
        raise RuntimeError(
            "MongoDB connection string is not configured. Create benchmark-runs/config.local.json."
        )
    return connection_string


def connect_to_db(db_config: JSON):
    mng = _require_pymongo()
    client = mng.MongoClient(_require_connection_string(db_config))
    database = client.get_database(str(db_config['database']))
    collection = database.get_collection(str(db_config['collection']))

    return client, database, collection


def write_to_db(collection, data: JSON):
    scheduler = data.get("scheduler") if isinstance(data, dict) else None
    run_id = scheduler.get("run_id") if isinstance(scheduler, dict) else None
    job_id = scheduler.get("job_id") if isinstance(scheduler, dict) else None
    if run_id and job_id:
        collection.replace_one(
            {
                "scheduler.run_id": run_id,
                "scheduler.job_id": job_id,
            },
            data,
            upsert=True,
        )
        return
    collection.insert_one(data)


def setup(config_filename="config.json"):
    config = read_config(config_filename)
    return connect_to_db(config['db'])
