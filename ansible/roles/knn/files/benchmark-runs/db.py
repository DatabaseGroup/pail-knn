import json
from typing import Mapping, List, Union, Tuple
import pymongo as mng
import pymongo.database
import pymongo.collection

# recursive type for json, see: https://github.com/python/typing/issues/182
JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]


def read_run(run_filename: str) -> JSON:
    with open(run_filename) as run_file:
        return json.loads(run_file.read())


def read_config(config_filename: str) -> JSON:
    with open(config_filename) as config_file:
        return json.loads(config_file.read())


def connect_to_db(db_config: JSON) -> Tuple[mng.MongoClient, mng.database.Database, mng.collection.Collection]:
    client = mng.MongoClient(db_config['connection_string'])
    database = client.get_database(db_config['database'])
    collection = database.get_collection(db_config['collection'])

    return client, database, collection


def write_to_db(collection: mng.collection.Collection, data: JSON):
    collection.insert_one(data)


def setup(config_filename="config.json") -> Tuple[mng.MongoClient, mng.database.Database, mng.collection.Collection]:
    config = read_config(config_filename)
    return connect_to_db(config['db'])
