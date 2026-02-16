import pymongo.collection


def get_by_label_dataset(collection: pymongo.collection.Collection, label: str, dataset: str):
    search_key = {
        "meta.label": label,
        "meta.dataset": dataset
    }

    results = collection.find_one(search_key)
    return results


def _get_default_preprocessing(key: str):
    return {
        '$addFields': {
            'aggregate': f'${key}'
        }
    }


def get_average(collection: pymongo.collection.Collection, prefix: str, dataset: str, group_key: str,
                result_group_key: str, key: str,
                result_key: str,
                preprocessing=None, additional_filter=None):
    if preprocessing is None:
        preprocessing = _get_default_preprocessing(key)

    filter = {
        '$match': {
            'meta.label': {"$regex": f"^{prefix}.*"},
            'meta.dataset': dataset,
            'meta.k': {'$lte': 80}
        }
    }

    if additional_filter is not None:
        filter['$match'].update(additional_filter)

    grouping = {
        'dataset': '$meta.dataset',
        'label': '$meta.label',
        result_group_key: group_key
    }

    pipeline = [
        filter,
        preprocessing,
        {
            '$group': {
                '_id': grouping,
                'avg': {
                    '$median': {'input': '$aggregate', 'method': 'approximate'}
                }
            }
        },
        {
            '$project': {
                '_id': 0,
                result_group_key: f'$_id.{result_group_key}',
                'label': '$_id.label',
                'avg': 1,
            }
        },
        {
            '$group': {
                '_id': f'${result_group_key}',
                'avgs': {
                    '$push': {
                        'k': '$label',
                        'v': '$avg'
                    }
                }
            }
        },
        {
            '$project': {
                '_id': 0,
                result_group_key: '$_id',
                f'{result_key}': {'$arrayToObject': '$avgs'}
            }
        },
        {
            '$sort': {
                result_group_key: 1
            }
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)


def get_average_by_k(collection: pymongo.collection.Collection, prefix: str, dataset: str, key: str, result_key: str,
                     preprocessing=None, additional_filter=None):
    group_key = '$meta.k'
    result_group_key = 'k'
    return get_average(collection, prefix, dataset, group_key, result_group_key, key, result_key,
                       preprocessing=preprocessing,
                       additional_filter=additional_filter)


def get_average_by_concurrency(collection: pymongo.collection.Collection, prefix: str, dataset: str, key: str,
                               result_key: str,
                               preprocessing=None, additional_filter=None):
    group_key = '$meta.concurrency'
    result_group_key = 'concurrency'
    return get_average(collection, prefix, dataset, group_key, result_group_key, key, result_key,
                       preprocessing=preprocessing,
                       additional_filter=additional_filter)
