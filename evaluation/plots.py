import pathlib
from collections import defaultdict

import config
import queries

import csv
import os

import pymongo.collection
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from typing import List, Dict, Any, Iterable, Union

PREFIX = "v10-"


def remove_extension(filename):
    return os.path.splitext(filename)[0]


def get_filename(dir: str, dataset: str, extension="csv"):
    return os.path.join(dir, f"{remove_extension(dataset)}.{extension}")


def strip_header_prefix(headers: list[str], prefix: str):
    return list(map(lambda h: h.removeprefix(prefix), headers))


def normalize(d: dict[str, float], normalizer: str, normalized: str):
    if normalizer in d and normalized in d:
        d[normalized] = d[normalized] / d[normalizer] - 1
    d[normalizer] = 0


def write_to_csv(filepath: str, headers: Iterable[str], rows: Iterable[Iterable]):
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def get_histogram(collection: pymongo.collection.Collection, dataset: str):
    res = queries.get_by_label_dataset(collection=collection, label=f"{PREFIX}slim", dataset=dataset)
    hist = res["statistics"]["index"]["group_histogram"]
    write_to_csv(get_filename("histograms", dataset), ("size", "count", "weight", "group"), hist)


def format_and_write_csv(outdir: str, outfile: str, result: list[dict[str, Union[int, dict[str, float]]]], key: str):
    labels = sorted(result[0]["result"].keys())
    headers = [key] + strip_header_prefix(labels, prefix=PREFIX)

    rows = []
    for item in result:
        row = [item[key]]
        row.extend([item["result"][label] if label in item["result"] else "" for label in labels])
        rows.append(row)

    write_to_csv(get_filename(outdir, outfile), headers, rows)


def get_average_by_k(collection: pymongo.collection.Collection, output: str, dataset: str, field: str,
                     preprocessing=None,
                     additional_filter=None):
    res = queries.get_average_by_k(collection=collection, prefix=PREFIX, dataset=dataset, key=field,
                                   result_key="result",
                                   preprocessing=preprocessing, additional_filter=additional_filter)

    format_and_write_csv(output, dataset, res, key="k")


def get_average_by_concurrency(collection: pymongo.collection.Collection, output: str, dataset: str, field: str,
                               preprocessing=None,
                               additional_filter=None):
    res = queries.get_average_by_concurrency(collection=collection, prefix=PREFIX, dataset=dataset, key=field,
                                             result_key="result",
                                             preprocessing=preprocessing, additional_filter=additional_filter)

    format_and_write_csv(output, dataset, res, key="concurrency")


def get_runtimes(collection: pymongo.collection.Collection, dataset: str):
    get_average_by_k(collection, "runtimes", dataset, "statistics.timing.join_time")


def get_throughput(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.concurrency': 1
    }

    preprocessing = {
        '$addFields': {
            'aggregate': {
                '$divide': ['$meta.sample_size', '$statistics.timing.join_time']
            }
        }
    }

    get_average_by_k(collection, "throughput", dataset, "statistics.timing.join_time", additional_filter=filter,
                     preprocessing=preprocessing)


def get_candidates(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.concurrency': 1
    }
    get_average_by_k(collection, "candidates", dataset, "statistics.join.avg_verifications.avg",
                     additional_filter=filter)


def get_k_similarity(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.label': f'{PREFIX}slim',
        'meta.concurrency': 1
    }
    get_average_by_k(collection, "ksimilarity", dataset, "statistics.join.final_k_similarity.avg",
                     additional_filter=filter)


def get_read_size(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.label': {"$regex": f"{PREFIX}(full|slim|eager|grouping|topkbaseline)"},
        'meta.concurrency': 1
    }
    get_average_by_k(collection, "readsize", dataset, "statistics.index.read_size.avg", additional_filter=filter)


def _average_results_list(results: list[dict]) -> dict:
    sums = defaultdict(float)
    num_items = len(results)

    for result_dict in results:
        for key, value in result_dict.items():
            sums[key] += value

    final_averages = {key: total / num_items for key, total in sums.items()}
    return final_averages


def get_avg_relative_data(collection: pymongo.collection.Collection, datasets: list[str], output: str,
                          query_func, key: str, group_key: str,
                          normalization_strategy: str, normalize_by: str = None,
                          additional_filter=None, preprocessing=None):
    """
    Generic function for computing relative data averaged across datasets.

    Args:
        query_func: queries.get_average_by_k or queries.get_average_by_concurrency
        group_key: 'k' or 'concurrency'
        normalization_strategy: 'by_algorithm' or 'by_baseline'
        normalize_by: algorithm name for 'by_algorithm' strategy
    """
    all_processed_data = []
    for dataset in datasets:
        raw_results = query_func(collection=collection, prefix=PREFIX, dataset=dataset,
                                 key=key, result_key="result",
                                 additional_filter=additional_filter, preprocessing=preprocessing)
        processed = [
            {group_key: d[group_key], 'result': {k.removeprefix(PREFIX): v for k, v in d['result'].items()}}
            for d in raw_results
        ]
        all_processed_data.append(processed)

    # Normalize data based on strategy
    all_normalized_data = []
    for dataset in all_processed_data:
        if normalization_strategy == 'by_algorithm':
            # Normalize by specified algorithm (like original get_avg_normalized)
            normalized_dataset = [
                {
                    group_key: item[group_key],
                    'result': {
                        alg: value / item['result'][normalize_by]
                        for alg, value in item['result'].items()
                    },
                }
                for item in dataset
            ]
        elif normalization_strategy == 'by_baseline':
            # Normalize by each algorithm's own baseline (like get_avg_relative_scalability)
            baseline_item = next((item for item in dataset if item[group_key] == 1), None)
            if not baseline_item:
                continue  # Skip datasets without baseline

            baseline_values = baseline_item['result']
            normalized_dataset = []
            for item in dataset:
                normalized_result = {}
                for algorithm, value in item['result'].items():
                    if algorithm in baseline_values and baseline_values[algorithm] > 0:
                        normalized_result[algorithm] = value / baseline_values[algorithm]
                    else:
                        normalized_result[algorithm] = 0

                normalized_dataset.append({
                    group_key: item[group_key],
                    'result': normalized_result
                })
        else:
            raise ValueError(f"Unknown normalization strategy: {normalization_strategy}")

        all_normalized_data.append(normalized_dataset)

    # Write individual dataset results
    for dataset_name, data in zip(datasets, all_normalized_data):
        format_and_write_csv(output, dataset_name, data, key=group_key)

    # Compute per-dataset averages
    suffix = "overk" if group_key == "k" else "overconcurrency"
    per_dataset_averages = []
    for dataset_name, normalized_dataset in zip(datasets, all_normalized_data):
        results_to_average = [item['result'] for item in normalized_dataset]
        avg_result = _average_results_list(results_to_average)

        per_dataset_averages.append({'dataset': dataset_name, 'result': avg_result})
        format_and_write_csv(output, f"{dataset_name}-{suffix}", [{group_key: 'avg', 'result': avg_result}], key=group_key)

    # Group by dimension and average across datasets
    grouped_by_dimension = defaultdict(list)
    for dataset in all_normalized_data:
        for item in dataset:
            grouped_by_dimension[item[group_key]].append(item['result'])

    cross_dataset_avg_by_dimension = []
    for dim_value, result_list in sorted(grouped_by_dimension.items()):
        avg_result = _average_results_list(result_list)
        cross_dataset_avg_by_dimension.append({group_key: dim_value, 'result': avg_result})

    all_suffix = "byk" if group_key == "k" else "byconcurrency"
    format_and_write_csv(output, f"all_{all_suffix}", cross_dataset_avg_by_dimension, key=group_key)

    # Compute total average
    results_for_total_average = [d['result'] for d in cross_dataset_avg_by_dimension]
    total_average_result = _average_results_list(results_for_total_average)

    total_key_value = 'total_avg'
    format_and_write_csv(output, "all_total", [{group_key: total_key_value, 'result': total_average_result}], key=group_key)


def get_avg_normalized(collection: pymongo.collection.Collection, datasets: list[str], key: str, normalize_by: str,
                       output: str, additional_filter=None):
    return get_avg_relative_data(
        collection=collection, datasets=datasets, output=output,
        query_func=queries.get_average_by_k, key=key, group_key='k',
        normalization_strategy='by_algorithm', normalize_by=normalize_by,
        additional_filter=additional_filter
    )


def get_scalability(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.k': 20
    }

    preprocessing = {
        '$addFields': {
            'aggregate': {
                '$divide': ['$meta.sample_size', '$statistics.timing.join_time']
            }
        }
    }

    get_average_by_concurrency(collection, "scalability", dataset, "statistics.timing.join_time",
                               additional_filter=filter,
                               preprocessing=preprocessing)


def get_relative_scalability(collection: pymongo.collection.Collection, dataset: str):
    filter = {
        'meta.k': 20
    }

    preprocessing = {
        '$addFields': {
            'aggregate': {
                '$divide': ['$meta.sample_size', '$statistics.timing.join_time']
            }
        }
    }

    # Get absolute scalability data
    res = queries.get_average_by_concurrency(collection=collection, prefix=PREFIX, dataset=dataset,
                                             key="statistics.timing.join_time", result_key="result",
                                             preprocessing=preprocessing, additional_filter=filter)

    # Find the 1-core baseline values for all algorithms
    baseline_item = next((r for r in res if r["concurrency"] == 1), None)
    if not baseline_item:
        return  # No baseline data available

    baseline_values = baseline_item["result"]

    # Normalize each concurrency level against the 1-core baseline
    normalized_res = []
    for item in res:
        concurrency = item["concurrency"]
        result = item["result"]

        # Create normalized result dict
        normalized_result = {}
        for algorithm, value in result.items():
            if algorithm in baseline_values and baseline_values[algorithm] > 0:
                normalized_result[algorithm] = value / baseline_values[algorithm]
            else:
                normalized_result[algorithm] = 0

        normalized_res.append({"concurrency": concurrency, "result": normalized_result})

    format_and_write_csv("scalability_relative", dataset, normalized_res, key="concurrency")


def get_avg_relative_scalability(collection: pymongo.collection.Collection, datasets: list[str], output: str):
    filter = {
        'meta.k': 20
    }

    preprocessing = {
        '$addFields': {
            'aggregate': {
                '$divide': ['$meta.sample_size', '$statistics.timing.join_time']
            }
        }
    }

    return get_avg_relative_data(
        collection=collection, datasets=datasets, output=output,
        query_func=queries.get_average_by_concurrency,
        key="statistics.timing.join_time", group_key='concurrency',
        normalization_strategy='by_baseline',
        additional_filter=filter, preprocessing=preprocessing
    )


def main():
    uri = config.db_config["connection_string"]

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi("1"))
    database = client.get_database(config.db_config["database"])
    collection = database.get_collection(config.db_config["collection"])

    set_datasets = [
        # "arxiv-all-modes-labels",
        "bms-pos-dedup-raw",
        "dblp-all-modes-labels-raw",
        # "dblpv14",
        # "github-all-modes-labels",
        "kosarak-dedup-raw",
        "livejournal-userswithgroups-dedup-raw",
        "lnonis1",
        "movies-all-modes-labels",
        "netflix",
        "orkut-userswithgroups-dedup-raw",
        "trackers",
        "twitter",
        # "twitter-all-modes-labels"
    ]

    single_thread_filter = {
        'meta.concurrency': 1
    }
    get_avg_normalized(collection, set_datasets, "statistics.timing.join_time", "slim", "throughput_relative",
                       additional_filter=single_thread_filter)
    get_avg_normalized(collection, set_datasets, "statistics.join.avg_verifications.avg", "slim", "candidates_relative",
                       additional_filter=single_thread_filter)

    # Compute relative scalability averaged across datasets
    get_avg_relative_scalability(collection, set_datasets, "scalability_relative")

    for dataset in set_datasets:
        get_histogram(collection, dataset)
        get_runtimes(collection, dataset)
        get_throughput(collection, dataset)
        get_candidates(collection, dataset)
        get_k_similarity(collection, dataset)
        get_read_size(collection, dataset)
        get_scalability(collection, dataset)
        # get_relative_scalability(collection, dataset)


if __name__ == "__main__":
    main()
