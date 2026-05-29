import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import config
import pymongo.collection
import queries
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_LABEL_PREFIX = "v18-"
K_VALUES = [5, 10, 20, 40, 80]
INDEX_BUILD_K = 5
SCALABILITY_K = 20
SCALABILITY_CONCURRENCIES = [2, 4, 8, 16, 32]
SCALABILITY_OUTPUT_CONCURRENCIES = [1, *SCALABILITY_CONCURRENCIES]

DEFAULT_DATASETS = [
    "bms-pos-dedup-raw",
    "dblp-all-modes-labels-raw",
    "friendster",
    "kosarak-dedup-raw",
    "livejournal-userswithgroups-dedup-raw",
    "lnonis1",
    "movies-all-modes-labels",
    "netflix",
    "orkut-userswithgroups-dedup-raw",
    "pmc-sentences",
    "trackers",
    "twitter",
]

QUERY_LABELS = [
    "linearscan",
    "batchedlinearscanpp",
    "full",
    "slim",
    "eager",
    "grouping",
    "topkbaseline",
    "puffinn",
    "partition",
    "palloc",
    "transformation",
    "les3",
]

SCALABILITY_LABELS = [
    "linearscan",
    "batchedlinearscanpp",
    "slim",
    "topkbaseline",
    "partition",
    "palloc",
    "transformation",
    "les3",
]

READ_SIZE_LABELS = [
    "full",
    "slim",
    "eager",
    "grouping",
    "topkbaseline",
    "partition",
    "palloc",
    "les3",
]

ALGORITHM_BY_LABEL = {
    "linearscan": "baseline",
    "batchedlinearscanpp": "baselinepp",
    "full": "full",
    "slim": "slim",
    "eager": "witheager",
    "grouping": "withgrouping",
    "topkbaseline": "topkbaseline",
    "puffinn": "puffinn",
    "partition": "partition",
    "palloc": "palloc",
    "transformation": "transformation",
    "les3": "les3",
}

RELATIVE_BASELINE_LABEL = "slim"
RELATIVE_MEAN_KINDS = ("arithmetic", "geometric")
STALE_RELATIVE_DIRS = ("slim_vs_others", "others_vs_slim")
STALE_SCALABILITY_OUTPUTS = ("scalability_relative",)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def remove_extension(filename: str) -> str:
    return Path(filename).stem


def get_filename(directory: str, dataset: str, extension: str = "csv") -> Path:
    return EVALUATION_DIR / directory / f"{remove_extension(dataset)}.{extension}"


def write_to_csv(filepath: Path, headers: Iterable[Any], rows: Iterable[Iterable[Any]]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def write_result_csv(
    outdir: str,
    outfile: str,
    results: list[dict[str, Any]],
    dimension_key: str,
    labels: list[str],
) -> None:
    headers = [dimension_key, *labels]
    rows = []
    for item in results:
        result = item.get("result", {})
        rows.append([item[dimension_key], *[result.get(label, "") for label in labels]])
    write_to_csv(get_filename(outdir, outfile), headers, rows)


def normalize_result_labels(
    results: list[dict[str, Any]],
    dimension_key: str,
    label_prefix: str,
) -> list[dict[str, Any]]:
    normalized = []
    for item in results:
        result = {
            label.removeprefix(label_prefix): value
            for label, value in item.get("result", {}).items()
            if value is not None
        }
        normalized.append({dimension_key: item[dimension_key], "result": result})
    return normalized


def standard_filter(k_values: Iterable[int] = K_VALUES) -> dict[str, Any]:
    return {
        "scheduler.experiment": "standard",
        "scheduler.requested_k": {"$in": list(k_values)},
        "meta.concurrency": 1,
    }


def selected_algorithm_filter(labels: list[str]) -> dict[str, Any]:
    return {"scheduler.selected_algorithm": {"$in": [ALGORITHM_BY_LABEL[label] for label in labels]}}


def merged_filter(*filters: dict[str, Any]) -> dict[str, Any]:
    return {"$and": list(filters)}


def throughput_preprocessing() -> dict[str, Any]:
    return {
        "$addFields": {
            "aggregate": {
                "$divide": ["$statistics.join.queries.performed", "$statistics.timing.join_time"],
            },
        },
    }


def average_by_k(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    dataset: str,
    field: str,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw = queries.get_average_by_k(
        collection=collection,
        prefix=label_prefix,
        dataset=dataset,
        key=field,
        result_key="result",
        preprocessing=preprocessing,
        additional_filter=additional_filter,
    )
    return normalize_result_labels(raw, "k", label_prefix)


def average_by_concurrency(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    dataset: str,
    field: str,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw = queries.get_average_by_concurrency(
        collection=collection,
        prefix=label_prefix,
        dataset=dataset,
        key=field,
        result_key="result",
        preprocessing=preprocessing,
        additional_filter=additional_filter,
    )
    return normalize_result_labels(raw, "concurrency", label_prefix)


def standard_metric_by_k(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    dataset: str,
    field: str,
    k_values: Iterable[int] = K_VALUES,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    filters = [standard_filter(k_values)]
    if additional_filter is not None:
        filters.append(additional_filter)
    return average_by_k(
        collection,
        label_prefix,
        dataset,
        field,
        preprocessing=preprocessing,
        additional_filter=merged_filter(*filters),
    )


def write_standard_metric(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    dataset: str,
    output: str,
    field: str,
    labels: list[str] = QUERY_LABELS,
    k_values: Iterable[int] = K_VALUES,
    preprocessing: dict[str, Any] | None = None,
    additional_filter: dict[str, Any] | None = None,
) -> None:
    result = standard_metric_by_k(
        collection,
        label_prefix,
        dataset,
        field,
        k_values=k_values,
        preprocessing=preprocessing,
        additional_filter=additional_filter,
    )
    write_result_csv(output, dataset, result, "k", labels)


def get_histogram(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    result = queries.get_by_label_dataset(
        collection=collection,
        label=f"{label_prefix}slim",
        dataset=dataset,
        additional_filter=merged_filter(standard_filter(), selected_algorithm_filter(["slim"])),
    )
    histogram = ((result or {}).get("statistics", {}).get("index", {}).get("group_histogram", []))
    if not histogram:
        warn(f"no histogram data for {dataset}")
    write_to_csv(get_filename("histograms", dataset), ("size", "count", "weight", "group"), histogram)


def get_runtimes(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    write_standard_metric(
        collection,
        label_prefix,
        dataset,
        "runtimes",
        "statistics.timing.join_time",
    )


def get_throughput(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    write_standard_metric(
        collection,
        label_prefix,
        dataset,
        "throughput",
        "statistics.timing.join_time",
        preprocessing=throughput_preprocessing(),
    )


def get_candidates(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    write_standard_metric(
        collection,
        label_prefix,
        dataset,
        "candidates",
        "statistics.join.avg_verifications.avg",
    )


def get_k_similarity(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    write_standard_metric(
        collection,
        label_prefix,
        dataset,
        "ksimilarity",
        "statistics.join.final_k_similarity.avg",
        labels=["slim"],
        additional_filter=selected_algorithm_filter(["slim"]),
    )


def get_read_size(collection: pymongo.collection.Collection, label_prefix: str, dataset: str) -> None:
    write_standard_metric(
        collection,
        label_prefix,
        dataset,
        "readsize",
        "statistics.index.read_size.avg",
        labels=READ_SIZE_LABELS,
        additional_filter=selected_algorithm_filter(READ_SIZE_LABELS),
    )


def scalability_filter() -> dict[str, Any]:
    return {
        "scheduler.requested_k": SCALABILITY_K,
        "$or": [
            {"scheduler.experiment": "standard", "meta.concurrency": 1},
            {
                "scheduler.experiment": "scalability",
                "meta.concurrency": {"$in": SCALABILITY_CONCURRENCIES},
            },
        ],
    }


def get_scalability(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    dataset: str,
) -> list[dict[str, Any]]:
    result = average_by_concurrency(
        collection,
        label_prefix,
        dataset,
        "statistics.timing.join_time",
        preprocessing=throughput_preprocessing(),
        additional_filter=merged_filter(scalability_filter(), selected_algorithm_filter(SCALABILITY_LABELS)),
    )
    write_result_csv("scalability", dataset, result, "concurrency", SCALABILITY_LABELS)
    return result


def mean_result_dicts(results: list[dict[str, float]], mean_kind: str, context: str) -> dict[str, float]:
    values_by_label: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for label, value in result.items():
            if mean_kind == "geometric" and value <= 0:
                warn(f"skipping {context} geometric mean for {label}: non-positive factor {value}")
                continue
            values_by_label[label].append(value)

    if mean_kind == "arithmetic":
        return {label: sum(values) / len(values) for label, values in values_by_label.items()}
    if mean_kind == "geometric":
        return {
            label: math.exp(sum(math.log(value) for value in values) / len(values))
            for label, values in values_by_label.items()
        }
    raise ValueError(f"unknown mean kind: {mean_kind}")


def average_result_dicts(results: list[dict[str, float]]) -> dict[str, float]:
    return mean_result_dicts(results, "arithmetic", "average")


def bytes_to_mb(value: float) -> float:
    return value / 2**20


def transform_values(values: dict[str, float], value_transform=None) -> dict[str, float]:
    if value_transform is None:
        return values
    return {label: value_transform(value) for label, value in values.items()}


def format_metric_values(values: dict[str, float], round_digits: int | None) -> list[Any]:
    if round_digits is None:
        return [values.get(label, "") for label in QUERY_LABELS]
    return ["" if label not in values else f"{values[label]:.{round_digits}f}" for label in QUERY_LABELS]


def relative_output_dir(output: str, mean_kind: str) -> str:
    return str(Path(output) / mean_kind)


def clear_relative_output(output: str) -> None:
    directory = EVALUATION_DIR / output
    if not directory.exists():
        return
    for path in directory.glob("*.csv"):
        path.unlink()
    for stale_dir in (*RELATIVE_MEAN_KINDS, *STALE_RELATIVE_DIRS):
        shutil.rmtree(directory / stale_dir, ignore_errors=True)


def relative_context(dataset: str, dimension_key: str, dimension: Any) -> str:
    return f"{dataset} {dimension_key}={dimension}"


def relative_summary_suffixes(dimension_key: str) -> tuple[str, str]:
    if dimension_key == "k":
        return "overk", "byk"
    if dimension_key == "concurrency":
        return "overconcurrency", "byconcurrency"
    if dimension_key == "interval":
        return "overinterval", "byinterval"
    return f"over{dimension_key}", f"by{dimension_key}"


def relative_dimension_sort_key(dimension_key: str, dimension: Any) -> tuple[Any, ...]:
    if dimension_key == "interval" and isinstance(dimension, str):
        start, separator, end = dimension.partition("-")
        if separator:
            try:
                return (0, int(start), int(end))
            except ValueError:
                pass
    if isinstance(dimension, (int, float)):
        return (0, dimension)
    return (1, str(dimension))


def build_relative_factor_rows(
    rows: list[dict[str, Any]],
    dimension_key: str,
    relative_to: str,
    output: str,
    dataset: str,
    metric_direction: str,
) -> list[dict[str, Any]]:
    relative_rows = []
    for row in rows:
        dimension = row[dimension_key]
        values = row.get("result", {})
        baseline = values.get(relative_to)
        context = relative_context(dataset, dimension_key, dimension)
        if baseline is None:
            warn(f"skipping {output} comparison for {context}: missing {relative_to}")
            continue
        if metric_direction == "lower" and baseline == 0:
            warn(f"skipping {output} comparison for {context}: {relative_to} is zero")
            continue

        result = {relative_to: 1.0}
        for label, value in values.items():
            if label == relative_to:
                continue
            if metric_direction == "higher":
                if value == 0:
                    warn(f"skipping {output} comparison for {context} {label}: denominator is zero")
                    continue
                result[label] = baseline / value
            elif metric_direction == "lower":
                result[label] = value / baseline
            else:
                raise ValueError(f"unknown metric direction: {metric_direction}")
        relative_rows.append({dimension_key: dimension, "result": result})
    return relative_rows


def write_relative_summaries(
    output: str,
    datasets: list[str],
    dimension_key: str,
    relative_by_dataset: dict[str, list[dict[str, Any]]],
    labels: list[str],
    mean_kind: str,
) -> None:
    item_suffix, all_suffix = relative_summary_suffixes(dimension_key)
    for dataset in datasets:
        rows = relative_by_dataset.get(dataset, [])
        write_result_csv(output, dataset, rows, dimension_key, labels)
        average = mean_result_dicts([row["result"] for row in rows], mean_kind, f"{output}/{dataset}-{item_suffix}")
        write_result_csv(
            output,
            f"{dataset}-{item_suffix}",
            [{dimension_key: "avg", "result": average}],
            dimension_key,
            labels,
        )

    grouped: dict[Any, list[dict[str, float]]] = defaultdict(list)
    for rows in relative_by_dataset.values():
        for row in rows:
            grouped[row[dimension_key]].append(row["result"])

    cross_dataset = [
        {
            dimension_key: dimension,
            "result": mean_result_dicts(results, mean_kind, f"{output}/all_{dimension}"),
        }
        for dimension, results in sorted(
            grouped.items(),
            key=lambda item: relative_dimension_sort_key(dimension_key, item[0]),
        )
    ]
    write_result_csv(output, f"all_{all_suffix}", cross_dataset, dimension_key, labels)

    total_average = mean_result_dicts(
        [row["result"] for rows in relative_by_dataset.values() for row in rows],
        mean_kind,
        f"{output}/all_total",
    )
    write_result_csv(
        output,
        "all_total",
        [{dimension_key: "total_avg", "result": total_average}],
        dimension_key,
        labels,
    )


def write_relative_mean_summaries(
    output: str,
    datasets: list[str],
    dimension_key: str,
    relative_by_dataset: dict[str, list[dict[str, Any]]],
    labels: list[str],
) -> None:
    clear_relative_output(output)
    for mean_kind in RELATIVE_MEAN_KINDS:
        write_relative_summaries(
            relative_output_dir(output, mean_kind),
            datasets,
            dimension_key,
            relative_by_dataset,
            labels,
            mean_kind,
        )


def get_avg_relative_standard(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    datasets: list[str],
    field: str,
    output: str,
    metric_direction: str,
    k_values: Iterable[int] = K_VALUES,
    preprocessing: dict[str, Any] | None = None,
) -> None:
    relative_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        rows = standard_metric_by_k(
            collection,
            label_prefix,
            dataset,
            field,
            k_values=k_values,
            preprocessing=preprocessing,
        )
        relative_by_dataset[dataset] = build_relative_factor_rows(
            rows, "k", RELATIVE_BASELINE_LABEL, output, dataset, metric_direction
        )

    write_relative_mean_summaries(
        output,
        datasets,
        "k",
        relative_by_dataset,
        QUERY_LABELS,
    )


def write_index_build_metric_matrix(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    datasets: list[str],
    output: str,
    field: str,
    relative_to: str | None = None,
    k: int = INDEX_BUILD_K,
    value_transform=None,
    round_digits: int | None = None,
) -> None:
    rows = []
    relative_results = []
    for dataset in datasets:
        result_rows = standard_metric_by_k(collection, label_prefix, dataset, field, k_values=[k])
        values = transform_values(result_rows[0]["result"] if result_rows else {}, value_transform)
        if relative_to is not None:
            baseline = values.get(relative_to)
            if baseline is None:
                warn(f"skipping {output} normalization for {dataset}: missing {relative_to}")
                values = {}
            elif baseline == 0:
                warn(f"skipping {output} normalization for {dataset}: {relative_to} is zero")
                values = {relative_to: 1.0}
            else:
                values = {
                    label: 1.0 if label == relative_to else value / baseline
                    for label, value in values.items()
                }
            relative_results.append(values)
        rows.append([dataset, *format_metric_values(values, round_digits)])

    if relative_to is not None:
        average = average_result_dicts(relative_results)
        rows.append(["avg", *format_metric_values(average, round_digits)])

    write_to_csv(EVALUATION_DIR / output, ["dataset", *QUERY_LABELS], rows)


def build_scalability_speedup_rows(rows: list[dict[str, Any]], output: str, dataset: str) -> list[dict[str, Any]]:
    by_concurrency = {row["concurrency"]: row.get("result", {}) for row in rows}
    baseline = by_concurrency.get(1, {})
    if not baseline:
        warn(f"skipping {output} for {dataset}: missing concurrency=1 baseline")
        return []

    speedup_rows = []
    for concurrency in SCALABILITY_OUTPUT_CONCURRENCIES:
        values = by_concurrency.get(concurrency)
        if values is None:
            continue
        context = relative_context(dataset, "concurrency", concurrency)

        result = {}
        for label, value in values.items():
            label_baseline = baseline.get(label)
            if label_baseline is None or label_baseline == 0:
                warn(f"skipping {output} for {context} {label}: missing concurrency=1 baseline")
                continue
            result[label] = value / label_baseline

        if result:
            speedup_rows.append({"concurrency": concurrency, "result": result})
    return speedup_rows


def build_scalability_interval_efficiency_rows(
    rows: list[dict[str, Any]],
    output: str,
    dataset: str,
) -> list[dict[str, Any]]:
    by_concurrency = {row["concurrency"]: row.get("result", {}) for row in rows}
    interval_rows = []
    for concurrency, next_concurrency in zip(
        SCALABILITY_OUTPUT_CONCURRENCIES,
        SCALABILITY_OUTPUT_CONCURRENCIES[1:],
    ):
        values = by_concurrency.get(concurrency)
        next_values = by_concurrency.get(next_concurrency)
        interval = f"{concurrency}-{next_concurrency}"
        context = relative_context(dataset, "interval", interval)
        if values is None or next_values is None:
            warn(f"skipping {output} for {context}: missing concurrency result")
            continue

        result = {}
        parallelism_factor = next_concurrency / concurrency
        for label, next_value in next_values.items():
            value = values.get(label)
            if value is None or value == 0:
                warn(f"skipping {output} for {context} {label}: missing lower-concurrency baseline")
                continue
            result[label] = (next_value / value) / parallelism_factor

        if result:
            interval_rows.append({"interval": interval, "result": result})
    return interval_rows


def get_avg_scalability_metrics(
    collection: pymongo.collection.Collection,
    label_prefix: str,
    datasets: list[str],
) -> None:
    for output in STALE_SCALABILITY_OUTPUTS:
        clear_relative_output(output)

    speedup_by_dataset: dict[str, list[dict[str, Any]]] = {}
    interval_efficiency_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        rows = average_by_concurrency(
            collection,
            label_prefix,
            dataset,
            "statistics.timing.join_time",
            preprocessing=throughput_preprocessing(),
            additional_filter=merged_filter(scalability_filter(), selected_algorithm_filter(SCALABILITY_LABELS)),
        )
        speedup_by_dataset[dataset] = build_scalability_speedup_rows(rows, "scalability_speedup", dataset)
        interval_efficiency_by_dataset[dataset] = build_scalability_interval_efficiency_rows(
            rows,
            "scalability_interval_efficiency",
            dataset,
        )

    write_relative_mean_summaries(
        "scalability_speedup",
        datasets,
        "concurrency",
        speedup_by_dataset,
        SCALABILITY_LABELS,
    )
    write_relative_mean_summaries(
        "scalability_interval_efficiency",
        datasets,
        "interval",
        interval_efficiency_by_dataset,
        SCALABILITY_LABELS,
    )


def connect_collection() -> tuple[MongoClient, pymongo.collection.Collection]:
    uri = config.db_config["connection_string"]
    if not uri:
        raise RuntimeError("MongoDB connection string is not configured in evaluation/config.py")
    client = MongoClient(uri, server_api=ServerApi("1"))
    database = client.get_database(config.db_config["database"])
    return client, database.get_collection(config.db_config["collection"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export evaluation CSVs from current benchmark MongoDB results.")
    parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX, help="Benchmark label prefix to plot")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client, collection = connect_collection()
    try:
        get_avg_relative_standard(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "statistics.timing.join_time",
            "throughput_relative",
            "higher",
            preprocessing=throughput_preprocessing(),
        )
        get_avg_relative_standard(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "statistics.join.avg_verifications.avg",
            "candidates_relative",
            "lower",
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "indexsize/summary.csv",
            "statistics.index.size_bytes",
            value_transform=bytes_to_mb,
            round_digits=3,
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "heappeak/summary.csv",
            "statistics.join.heap_peak_bytes.max",
            value_transform=bytes_to_mb,
            round_digits=3,
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "indexingtime/summary.csv",
            "statistics.timing.indexing_time",
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "indexsize/relative.csv",
            "statistics.index.size_bytes",
            relative_to="slim",
            value_transform=bytes_to_mb,
            round_digits=3,
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "heappeak/relative.csv",
            "statistics.join.heap_peak_bytes.max",
            relative_to="slim",
            value_transform=bytes_to_mb,
            round_digits=3,
        )
        write_index_build_metric_matrix(
            collection,
            args.label_prefix,
            DEFAULT_DATASETS,
            "indexingtime/relative.csv",
            "statistics.timing.indexing_time",
            relative_to="slim",
        )
        get_avg_scalability_metrics(collection, args.label_prefix, DEFAULT_DATASETS)

        for dataset in DEFAULT_DATASETS:
            get_histogram(collection, args.label_prefix, dataset)
            get_runtimes(collection, args.label_prefix, dataset)
            get_throughput(collection, args.label_prefix, dataset)
            get_candidates(collection, args.label_prefix, dataset)
            get_k_similarity(collection, args.label_prefix, dataset)
            get_read_size(collection, args.label_prefix, dataset)
            get_scalability(collection, args.label_prefix, dataset)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
