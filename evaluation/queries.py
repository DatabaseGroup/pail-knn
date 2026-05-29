import re
from typing import Any

import pymongo.collection


def _successful_query_filter(prefix: str, dataset: str, key: str | None = None) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {"state": "success"},
        {"scheduler.phase": "query"},
        {"meta.dataset": dataset},
        {"meta.label": {"$regex": f"^{re.escape(prefix)}"}},
    ]
    if key is not None:
        conditions.append({key: {"$exists": True}})
    return conditions


def get_by_label_dataset(
    collection: pymongo.collection.Collection,
    label: str,
    dataset: str,
    additional_filter: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    match_conditions = [
        {"state": "success"},
        {"scheduler.phase": "query"},
        {"meta.label": label},
        {"meta.dataset": dataset},
    ]
    if additional_filter is not None:
        match_conditions.append(additional_filter)

    return collection.find_one(
        {"$and": match_conditions},
        sort=[("scheduler.requested_k", 1), ("meta.date", -1)],
    )


def _get_default_preprocessing(key: str) -> dict[str, Any]:
    return {"$addFields": {"aggregate": f"${key}"}}


def get_average(
    collection: pymongo.collection.Collection,
    prefix: str,
    dataset: str,
    group_key: str,
    result_group_key: str,
    key: str,
    result_key: str,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    preprocessing = preprocessing or _get_default_preprocessing(key)

    match_conditions = _successful_query_filter(prefix, dataset, key)
    if additional_filter is not None:
        match_conditions.append(additional_filter)

    pipeline = [
        {"$match": {"$and": match_conditions}},
        preprocessing,
        {"$match": {"aggregate": {"$ne": None}}},
        {
            "$group": {
                "_id": {
                    "label": "$meta.label",
                    result_group_key: group_key,
                },
                "avg": {"$median": {"input": "$aggregate", "method": "approximate"}},
            }
        },
        {
            "$project": {
                "_id": 0,
                result_group_key: f"$_id.{result_group_key}",
                "label": "$_id.label",
                "avg": 1,
            }
        },
        {
            "$group": {
                "_id": f"${result_group_key}",
                "avgs": {
                    "$push": {
                        "k": "$label",
                        "v": "$avg",
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                result_group_key: "$_id",
                result_key: {"$arrayToObject": "$avgs"},
            }
        },
        {"$sort": {result_group_key: 1}},
    ]

    return list(collection.aggregate(pipeline))


def get_average_by_k(
    collection: pymongo.collection.Collection,
    prefix: str,
    dataset: str,
    key: str,
    result_key: str,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return get_average(
        collection,
        prefix,
        dataset,
        "$scheduler.requested_k",
        "k",
        key,
        result_key,
        preprocessing=preprocessing,
        additional_filter=additional_filter,
    )


def get_average_by_concurrency(
    collection: pymongo.collection.Collection,
    prefix: str,
    dataset: str,
    key: str,
    result_key: str,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return get_average(
        collection,
        prefix,
        dataset,
        "$meta.concurrency",
        "concurrency",
        key,
        result_key,
        preprocessing=preprocessing,
        additional_filter=additional_filter,
    )
