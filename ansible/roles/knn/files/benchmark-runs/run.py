#!/bin/env python3
import json.decoder
import logging
import sys
import subprocess

import pymongo.database
import pymongo.collection
from bson import json_util

from db import *
import config

# globals
mongo_client: mng.MongoClient
mongo_database: mng.database.Database
mongo_collection: mng.collection.Collection
logging.basicConfig()
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


def main():
    global mongo_client, mongo_database, mongo_collection
    mongo_client, mongo_database, mongo_collection = connect_to_db(config.db_config)

    try:
        results = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=8*60*60) # 8 hour timeout
        result = json_util.loads(results.stdout)
        write_to_db(mongo_collection, result)
    except subprocess.TimeoutExpired:
        logger.info("Timeout for {}".format(" ".join(sys.argv[1:])))
    except json.decoder.JSONDecodeError:
        logger.warning("Could not decode executable output {}".format(sys.argv[1:]))
        logger.warning("{}".format(results.stdout))


if __name__ == "__main__":
    main()
