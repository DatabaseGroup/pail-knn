import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset


INITIAL_GROUPS = 128
PARTITION_THRESHOLD = 50
EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
TARGET_GROUP_FRACTION = 0.005
TRAINING_SAMPLE_TARGET = 200
PREDICTION_BATCH_SIZE = 8192
TOKEN_REPRESENTATION_MATRIX_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


class Parameters:
    epoch = EPOCHS
    n_neurons = 8
    output_dim = 1


class SiameseDataset(Dataset):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        left = torch.FloatTensor([float(i) for i in str(self.X[idx][0]).split()])
        right = torch.FloatTensor([float(i) for i in str(self.X[idx][1]).split()])
        force = torch.FloatTensor(self.labels[idx])

        return (left, right), force


class TensorPairDataset(Dataset):
    def __init__(self, left, right, labels):
        self.left = torch.as_tensor(left, dtype=torch.float32)
        self.right = torch.as_tensor(right, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.left[idx], self.right[idx]), self.labels[idx]


class ContrastiveLoss(nn.Module):
    def __init__(self, inV=1):
        super(ContrastiveLoss, self).__init__()
        self.aplha = inV

    def forward(self, output1, output2, force, left_sizes=None, right_sizes=None):
        output1 = output1.type(torch.FloatTensor)
        output2 = output2.type(torch.FloatTensor)
        force = force.type(torch.FloatTensor)

        tensor_0_5 = torch.full_like(output1, 0.5)
        left_0 = torch.le(output1, tensor_0_5)
        right_0 = torch.le(output2, tensor_0_5)

        common_group = torch.eq(left_0, right_0).type(torch.FloatTensor)

        tensor_diff = (tensor_0_5 - (output1 - output2).abs()).type(torch.FloatTensor)
        tensor_diff = tensor_diff * common_group

        loss = force * tensor_diff

        return loss.mean()


class EmbeddingNetMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingNetMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Parameters.n_neurons, bias=True),
            nn.ReLU(),
            nn.Linear(Parameters.n_neurons, Parameters.n_neurons, bias=True),
            nn.ReLU(),
            nn.Linear(Parameters.n_neurons, output_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.net(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


def available_cpu_cores():
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def configure_torch_threads():
    threads = available_cpu_cores()
    torch.set_num_threads(threads)
    return threads


def fit_siamese_tensors(
    left,
    right,
    labels,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    start_epoch=0,
    batch_size=BATCH_SIZE,
):
    left = torch.as_tensor(left, dtype=torch.float32)
    right = torch.as_tensor(right, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)

    for _ in range(0, start_epoch):
        scheduler.step()
    for _ in range(start_epoch, n_epochs):
        train_siamese_tensors(left, right, labels, model, loss_fn, optimizer, batch_size)
        scheduler.step()
        Parameters.epoch += 1


def train_siamese_tensors(left, right, labels, model, loss_fn, optimizer, batch_size):
    model.train()
    sample_count = len(labels)
    if sample_count == 0:
        return 0

    permutation = torch.randperm(sample_count)
    left = left[permutation]
    right = right[permutation]
    labels = labels[permutation]

    total_loss = 0
    batch_count = 0
    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)

        optimizer.zero_grad()
        outputs = model(left[batch_start:batch_end], right[batch_start:batch_end])

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_outputs = loss_fn(*(outputs + (labels[batch_start:batch_end],)))
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        batch_count += 1

    return total_loss / batch_count


def current_timestamp_json():
    return {"$date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def parse_set_line(line, line_no):
    try:
        tokens = [int(v) for v in line.strip().split()]
    except ValueError as error:
        raise ValueError(f"Invalid token id at line {line_no}") from error
    for token in tokens:
        if token < 0:
            raise ValueError(f"LES3 precompute requires non-negative token ids, found {token} at line {line_no}")
    return tokens


def parse_ptr_line(line, line_no):
    try:
        set_id_text, vector = line.strip().split(",", maxsplit=1)
        return int(set_id_text), vector.strip()
    except ValueError as error:
        raise ValueError(f"Invalid PTR representation at line {line_no}") from error


def scan_set_file(path):
    cardinality = 0
    max_token = -1
    min_tokens = []
    offsets = []
    with Path(path).open("rb") as read_file:
        while True:
            offset = read_file.tell()
            line = read_file.readline()
            if not line:
                break
            tokens = parse_set_line(line.decode("utf-8"), cardinality + 1)
            offsets.append(offset)
            min_tokens.append(min(tokens) if tokens else math.inf)
            for token in tokens:
                max_token = max(max_token, token)
            cardinality += 1
    return cardinality, max_token + 1, min_tokens, offsets


def scan_set_file_metadata(path):
    cardinality = 0
    max_token = -1
    with Path(path).open("r", encoding="utf-8") as read_file:
        for line_no, line in enumerate(read_file, start=1):
            tokens = parse_set_line(line, line_no)
            for token in tokens:
                max_token = max(max_token, token)
            cardinality += 1
    return cardinality, max_token + 1


def build_ptr_offsets(path, expected_count):
    offsets = [None] * expected_count
    dimensions = None
    with Path(path).open("rb") as read_file:
        line_no = 0
        while True:
            offset = read_file.tell()
            line = read_file.readline()
            if not line:
                break
            line_no += 1
            if not line.strip():
                continue
            set_id, vector = parse_ptr_line(line.decode("utf-8"), line_no)
            if set_id < 0 or set_id >= expected_count:
                raise ValueError(f"Out-of-range set id in PTR representation at line {line_no}")
            if offsets[set_id] is not None:
                raise ValueError(f"Duplicate set id in PTR representation: {set_id}")
            offsets[set_id] = offset
            current_dimensions = len(vector.split()) if vector else 0
            if dimensions is None:
                dimensions = current_dimensions
            elif dimensions != current_dimensions:
                raise ValueError(f"Inconsistent PTR vector dimension at line {line_no}")

    missing = [set_id for set_id, offset in enumerate(offsets) if offset is None]
    if missing:
        raise ValueError(f"PTR representation file is missing set id {missing[0]}")
    return offsets, dimensions or 0


def read_set_at(read_file, offsets, set_id):
    read_file.seek(offsets[set_id])
    return parse_set_line(read_file.readline().decode("utf-8"), set_id + 1)


def read_ptr_at(read_file, offsets, set_id):
    read_file.seek(offsets[set_id])
    found_set_id, vector = parse_ptr_line(read_file.readline().decode("utf-8"), set_id + 1)
    if found_set_id != set_id:
        raise ValueError(f"PTR offset index mismatch for set id {set_id}")
    return f"{set_id},{vector}"


def read_ptr_vector_at(read_file, offsets, set_id, dimensions):
    read_file.seek(offsets[set_id])
    found_set_id, vector = parse_ptr_line(read_file.readline().decode("utf-8"), set_id + 1)
    if found_set_id != set_id:
        raise ValueError(f"PTR offset index mismatch for set id {set_id}")

    if not vector:
        values = np.zeros(0, dtype=np.float32)
    else:
        values = np.fromstring(vector, sep=" ", dtype=np.float32)
    if values.size != dimensions:
        raise ValueError(f"PTR vector dimension mismatch for set id {set_id}")
    return values


def read_set_map(read_file, offsets, set_ids):
    sets = {}
    for set_id in sorted(set(set_ids), key=lambda value: offsets[value]):
        sets[set_id] = read_set_at(read_file, offsets, set_id)
    return sets


def read_ptr_vector_map(read_file, offsets, set_ids, dimensions):
    vectors = {}
    for set_id in sorted(set(set_ids), key=lambda value: offsets[value]):
        vectors[set_id] = read_ptr_vector_at(read_file, offsets, set_id, dimensions)
    return vectors


def read_ptr_vectors(read_file, offsets, set_ids, dimensions):
    if not set_ids:
        return np.zeros((0, dimensions), dtype=np.float32)

    vectors = [None] * len(set_ids)
    indexed_ids = list(enumerate(set_ids))
    for index, set_id in sorted(indexed_ids, key=lambda item: offsets[item[1]]):
        vectors[index] = read_ptr_vector_at(read_file, offsets, set_id, dimensions)
    return np.vstack(vectors).astype(np.float32, copy=False)


def ptr_dimensions(num_tokens):
    if num_tokens <= 1:
        return 0
    return 2 * int(math.ceil(math.log(num_tokens, 2)))


def token_representation_matrix(num_tokens, dimensions):
    if num_tokens <= 0 or dimensions == 0:
        return np.zeros((num_tokens, 0), dtype=np.uint8)

    matrix = np.zeros((num_tokens, dimensions), dtype=np.uint8)
    mid_value = num_tokens / 2
    iterations = dimensions // 2
    for bit in range(iterations):
        denominator = mid_value / math.pow(2, bit)
        indicators = (np.floor(np.arange(num_tokens, dtype=np.float64) / denominator).astype(np.int64) % 2)
        matrix[:, 2 * bit] = indicators == 0
        matrix[:, 2 * bit + 1] = indicators == 1
    return matrix


def add_token_representation(target, token, num_tokens):
    if target.size == 0:
        return
    mid_value = num_tokens / 2
    for bit in range(target.size // 2):
        indicator = int(math.floor(token / (mid_value / math.pow(2, bit)))) % 2
        target[(2 * bit) + indicator] += 1


def set_representation(tokens, token_matrix, num_tokens, dimensions):
    if dimensions == 0:
        return np.zeros(0, dtype=np.int64)
    if token_matrix is not None:
        if not tokens:
            return np.zeros(dimensions, dtype=np.int64)
        return token_matrix[np.asarray(tokens, dtype=np.int64)].sum(axis=0, dtype=np.int64)

    rep = np.zeros(dimensions, dtype=np.int64)
    for token in tokens:
        add_token_representation(rep, token, num_tokens)
    return rep


def can_materialize_token_matrix(num_tokens, dimensions):
    return num_tokens * dimensions <= TOKEN_REPRESENTATION_MATRIX_LIMIT_BYTES


def stream_representations(input_path, output_path):
    cardinality, num_tokens = scan_set_file_metadata(input_path)
    dimensions = ptr_dimensions(num_tokens)
    token_matrix = (
        token_representation_matrix(num_tokens, dimensions)
        if can_materialize_token_matrix(num_tokens, dimensions)
        else None
    )

    max_rep_num = 1
    with Path(input_path).open("r", encoding="utf-8") as read_file:
        for line_no, line in enumerate(read_file, start=1):
            rep = set_representation(parse_set_line(line, line_no), token_matrix, num_tokens, dimensions)
            if rep.size:
                max_rep_num = max(max_rep_num, int(rep.max()))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    with Path(input_path).open("r", encoding="utf-8") as read_file, tmp_output.open(
        "w",
        encoding="utf-8",
    ) as write_file:
        for set_id, line in enumerate(read_file):
            rep = set_representation(parse_set_line(line, set_id + 1), token_matrix, num_tokens, dimensions)
            normalized = rep.astype(np.float64) / max_rep_num if rep.size else rep.astype(np.float64)
            write_file.write(str(set_id) + "," + " ".join(str(value) for value in normalized) + "\n")
    tmp_output.replace(output)

    return cardinality, num_tokens, dimensions


def partition_based_rep(numTokens):
    if numTokens <= 0:
        return []

    matrix = token_representation_matrix(numTokens, ptr_dimensions(numTokens))
    return [row.tolist() for row in matrix]


def transaction_representation(transaction, reps_of_tokens):
    if not reps_of_tokens:
        return []

    rep = []
    for dim in range(len(reps_of_tokens[0])):
        count = 0
        for token in transaction:
            if reps_of_tokens[token][dim] == 1:
                count += 1
        rep.append(count)
    return rep


def compute_representations(database, numTokens):
    dimensions = ptr_dimensions(numTokens)
    token_matrix = token_representation_matrix(numTokens, dimensions)
    max_rep_num = 1
    raw_reps = []
    for set_tokens in database:
        rep = set_representation(set_tokens, token_matrix, numTokens, dimensions)
        raw_reps.append(rep)
        if rep.size:
            max_rep_num = max(max_rep_num, int(rep.max()))

    representations = []
    for set_id, rep in enumerate(raw_reps):
        normalized = rep.astype(np.float64) / max_rep_num if rep.size else rep.astype(np.float64)
        representations.append(str(set_id) + "," + " ".join(str(value) for value in normalized))
    return representations


def initialize_groups(partition_order, num_init_groups):
    if len(partition_order) == 0:
        return []
    avg_group_size = int(len(partition_order) / num_init_groups)
    if avg_group_size <= 0:
        raise ValueError("INITIAL_GROUPS must not exceed the number of sets")

    groups = []
    current_group = []
    for set_id in partition_order:
        current_group.append(set_id)
        if len(current_group) == avg_group_size:
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)
    return groups


def target_group_count(cardinality):
    if cardinality <= 0:
        return 0
    return max(1, int(math.ceil(TARGET_GROUP_FRACTION * cardinality)))


def deduce_level_count(cardinality, initial_group_count):
    target = target_group_count(cardinality)
    if cardinality <= 0 or initial_group_count <= 0 or initial_group_count >= target:
        return 0, target
    return int(math.ceil(math.log2(target / initial_group_count))), target


def compute_dist(point1, point2):
    common_tokens = set(point1).intersection(set(point2))
    denominator = len(point1) + len(point2)
    if denominator == 0:
        return 0.0
    result = 1.0 - 2.0 * len(common_tokens) / denominator
    return result


def group_rows(group, set_file, ptr_file, set_offsets, ptr_offsets):
    rows = []
    for set_id in group:
        rows.append((set_id, read_ptr_at(ptr_file, ptr_offsets, set_id), read_set_at(set_file, set_offsets, set_id)))
    return rows


def sample_training_pairs(group):
    if len(group) < PARTITION_THRESHOLD:
        return []
    if len(group) <= TRAINING_SAMPLE_TARGET:
        return [(left_id, right_id) for left_id in group for right_id in group if left_id != right_id]

    left_ids = random.sample(group, TRAINING_SAMPLE_TARGET)
    right_ids = random.sample(group, TRAINING_SAMPLE_TARGET)
    return [(left_id, right_id) for left_id in left_ids for right_id in right_ids if left_id != right_id]


def compute_dist_from_prepared(left, right):
    left_tokens, left_size = left
    right_tokens, right_size = right
    denominator = left_size + right_size
    if denominator == 0:
        return 0.0
    return 1.0 - 2.0 * len(left_tokens.intersection(right_tokens)) / denominator


def prepare_training_data(pairs, set_tokens, ptr_vectors, dimensions):
    left_vectors = np.empty((len(pairs), dimensions), dtype=np.float32)
    right_vectors = np.empty((len(pairs), dimensions), dtype=np.float32)
    labels = np.empty((len(pairs), 1), dtype=np.float32)
    prepared_sets = {set_id: (set(tokens), len(tokens)) for set_id, tokens in set_tokens.items()}
    all_sets_identical = True

    for index, (left_id, right_id) in enumerate(pairs):
        distance = compute_dist_from_prepared(prepared_sets[left_id], prepared_sets[right_id])
        if distance != 0:
            all_sets_identical = False
        left_vectors[index] = ptr_vectors[left_id]
        right_vectors[index] = ptr_vectors[right_id]
        labels[index, 0] = distance

    return left_vectors, right_vectors, labels, all_sets_identical


def generate_training_samples(rows):
    all_sets_identical = True
    training_samples = []
    training_labels = []
    row_by_id = {set_id: row for set_id, *row in rows}
    for left_id, right_id in sample_training_pairs([row[0] for row in rows]):
        left_representation, left_tokens = row_by_id[left_id]
        right_representation, right_tokens = row_by_id[right_id]
        two_set_dist = compute_dist(left_tokens, right_tokens)

        if two_set_dist != 0:
            all_sets_identical = False
        training_samples.append([left_representation.split(",")[1], right_representation.split(",")[1]])
        training_labels.append([two_set_dist])
    return training_samples, training_labels, all_sets_identical


def split_randomly(first_set, second_set):
    all_items = []
    if len(first_set) == 0:
        all_items = second_set.copy()
    elif len(second_set) == 0:
        all_items = first_set.copy()
    if len(first_set) == 0 or len(second_set) == 0:
        first_set.clear()
        second_set.clear()
        for item in all_items:
            if random.uniform(0, 1) < 0.5:
                first_set.append(item)
            else:
                second_set.append(item)


def empty_partition_timing():
    return {
        "sampling_wall": 0.0,
        "set_reading_wall": 0.0,
        "ptr_reading_wall": 0.0,
        "training_wall": 0.0,
        "prediction_wall": 0.0,
    }


def add_partition_timing(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + value


def split_group(group, set_file, ptr_file, set_offsets, ptr_offsets, ptr_vector_dimensions, timing=None):
    if len(group) < PARTITION_THRESHOLD:
        return [group]
    if ptr_vector_dimensions == 0:
        return [group]

    timing = timing if timing is not None else empty_partition_timing()

    sampling_start = time.perf_counter()
    training_pairs = sample_training_pairs(group)
    timing["sampling_wall"] += time.perf_counter() - sampling_start
    if not training_pairs:
        return [group]

    required_set_ids = {set_id for pair in training_pairs for set_id in pair}

    set_reading_start = time.perf_counter()
    set_tokens = read_set_map(set_file, set_offsets, required_set_ids)
    timing["set_reading_wall"] += time.perf_counter() - set_reading_start

    ptr_reading_start = time.perf_counter()
    sampled_ptr_vectors = read_ptr_vector_map(ptr_file, ptr_offsets, required_set_ids, ptr_vector_dimensions)
    timing["ptr_reading_wall"] += time.perf_counter() - ptr_reading_start

    left_vectors, right_vectors, training_labels, all_sets_identical = prepare_training_data(
        training_pairs,
        set_tokens,
        sampled_ptr_vectors,
        ptr_vector_dimensions,
    )
    if all_sets_identical:
        return [group]

    loss_fn = ContrastiveLoss()

    input_dim = ptr_vector_dimensions

    embedding_net = EmbeddingNetMLP(input_dim, 1)
    model = SiameseNet(embedding_net)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    training_start = time.perf_counter()
    fit_siamese_tensors(
        left_vectors,
        right_vectors,
        training_labels,
        model,
        loss_fn,
        optimizer,
        scheduler,
        EPOCHS,
    )
    timing["training_wall"] += time.perf_counter() - training_start

    avg_prediction = 0.5
    first_set = []
    second_set = []
    model.eval()
    for chunk_start in range(0, len(group), PREDICTION_BATCH_SIZE):
        chunk_ids = group[chunk_start:chunk_start + PREDICTION_BATCH_SIZE]

        ptr_reading_start = time.perf_counter()
        chunk_vectors = read_ptr_vectors(ptr_file, ptr_offsets, chunk_ids, ptr_vector_dimensions)
        timing["ptr_reading_wall"] += time.perf_counter() - ptr_reading_start

        prediction_start = time.perf_counter()
        with torch.no_grad():
            predictions = model.get_embedding(torch.from_numpy(chunk_vectors)).reshape(-1).numpy()
        timing["prediction_wall"] += time.perf_counter() - prediction_start

        for set_id, model_group in zip(chunk_ids, predictions):
            if model_group > avg_prediction:
                first_set.append(set_id)
            else:
                second_set.append(set_id)

    split_randomly(first_set, second_set)

    result = []
    if len(first_set) > 0:
        result.append(first_set)
    if len(second_set) > 0:
        result.append(second_set)
    return result


def split_level(groups, input_path, ptr_path, set_offsets, ptr_offsets, ptr_vector_dimensions):
    cpu_start = time.process_time()
    timing = empty_partition_timing()
    next_groups = []
    with Path(input_path).open("rb") as set_file, Path(ptr_path).open("rb") as ptr_file:
        for group in groups:
            next_groups.extend(
                split_group(group, set_file, ptr_file, set_offsets, ptr_offsets, ptr_vector_dimensions, timing)
            )
    return next_groups, time.process_time() - cpu_start, timing


def write_groups(groups, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as write_file:
        write_group_lines(write_file, groups)
    tmp_output.replace(output)


def write_group_lines(write_file, groups):
    for group in groups:
        write_file.write(" ".join(str(set_id) for set_id in group) + "\n")


def read_groups(path, cardinality):
    groups = []
    seen = [False] * cardinality
    with Path(path).open("r", encoding="utf-8") as read_file:
        for line_no, line in enumerate(read_file, start=1):
            line = line.strip()
            if not line:
                continue
            group = []
            for value in line.split():
                try:
                    set_id = int(value)
                except ValueError as error:
                    raise ValueError(f"Invalid record id in LES3 group file at line {line_no}") from error
                if set_id < 0 or set_id >= cardinality:
                    raise ValueError(f"Out-of-range record id in LES3 group file at line {line_no}")
                if seen[set_id]:
                    raise ValueError(f"Duplicate record id in LES3 group file: {set_id}")
                seen[set_id] = True
                group.append(set_id)
            if group:
                groups.append(group)

    missing = [set_id for set_id, value in enumerate(seen) if not value]
    if missing:
        raise ValueError(f"Missing record id in LES3 group file: {missing[0]}")
    return groups


def read_partial_groups(path, cardinality, max_lines=None):
    groups = []
    seen = [False] * cardinality
    line_count = 0
    with Path(path).open("r", encoding="utf-8") as read_file:
        for line_no, line in enumerate(read_file, start=1):
            if max_lines is not None and line_count >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            group = []
            for value in line.split():
                try:
                    set_id = int(value)
                except ValueError as error:
                    raise ValueError(f"Invalid record id in LES3 partial group file at line {line_no}") from error
                if set_id < 0 or set_id >= cardinality:
                    raise ValueError(f"Out-of-range record id in LES3 partial group file at line {line_no}")
                if seen[set_id]:
                    raise ValueError(f"Duplicate record id in LES3 partial group file: {set_id}")
                seen[set_id] = True
                group.append(set_id)
            if group:
                groups.append(group)
                line_count += 1

    if max_lines is not None and line_count < max_lines:
        raise ValueError(f"LES3 partial group file has {line_count} complete groups, expected {max_lines}")
    return groups


def group_file_base(input_path):
    return Path(input_path).name + ".groups"


def level_output_path(output_dir, input_path, level):
    return Path(output_dir) / f"{group_file_base(input_path)}.level-{level:03d}"


def in_progress_level_output_path(output_dir, input_path, level):
    path = level_output_path(output_dir, input_path, level)
    return path.with_name(path.name + ".in-progress")


def level_progress_path(output_dir, input_path, level):
    path = level_output_path(output_dir, input_path, level)
    return path.with_name(path.name + ".progress.json")


def final_output_path(output_dir, input_path):
    return Path(output_dir) / group_file_base(input_path)


def copy_final_level(level_path, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = final_path.with_name(final_path.name + ".tmp")
    shutil.copyfile(level_path, tmp_final)
    tmp_final.replace(final_path)


def read_level_progress(path):
    path = Path(path)
    if not path.exists():
        return {"completed_parent_groups": 0, "output_group_lines": 0}
    with path.open("r", encoding="utf-8") as read_file:
        progress = json.load(read_file)
    return {
        "completed_parent_groups": int(progress.get("completed_parent_groups", 0)),
        "output_group_lines": int(progress.get("output_group_lines", 0)),
    }


def write_level_progress(path, completed_parent_groups, output_group_lines):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as write_file:
        json.dump(
            {
                "completed_parent_groups": completed_parent_groups,
                "output_group_lines": output_group_lines,
            },
            write_file,
        )
        write_file.write("\n")
    tmp_path.replace(path)


def load_in_progress_level(in_progress_path, progress_path, cardinality):
    if not Path(in_progress_path).exists():
        return 0, []

    try:
        progress = read_level_progress(progress_path)
        completed_parent_groups = progress["completed_parent_groups"]
        output_group_lines = progress["output_group_lines"]
        partial_groups = read_partial_groups(in_progress_path, cardinality, output_group_lines)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0, []

    write_groups(partial_groups, in_progress_path)
    return completed_parent_groups, partial_groups


def split_level_resumable(
    groups,
    input_path,
    ptr_path,
    set_offsets,
    ptr_offsets,
    ptr_vector_dimensions,
    output_dir,
    level,
    cardinality,
):
    output_path = level_output_path(output_dir, input_path, level)
    in_progress_path = in_progress_level_output_path(output_dir, input_path, level)
    progress_path = level_progress_path(output_dir, input_path, level)

    completed_parent_groups, next_groups = load_in_progress_level(in_progress_path, progress_path, cardinality)
    if completed_parent_groups > len(groups):
        completed_parent_groups = 0
        next_groups = []
        write_groups(next_groups, in_progress_path)

    cpu_start = time.process_time()
    timing = empty_partition_timing()
    in_progress_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if completed_parent_groups else "w"

    with Path(input_path).open("rb") as set_file, Path(ptr_path).open("rb") as ptr_file, in_progress_path.open(
        file_mode,
        encoding="utf-8",
    ) as write_file:
        for group_index in range(completed_parent_groups, len(groups)):
            split_groups = split_group(
                groups[group_index],
                set_file,
                ptr_file,
                set_offsets,
                ptr_offsets,
                ptr_vector_dimensions,
                timing,
            )
            write_group_lines(write_file, split_groups)
            next_groups.extend(split_groups)
            write_file.flush()
            os.fsync(write_file.fileno())
            write_level_progress(progress_path, group_index + 1, len(next_groups))

    in_progress_path.replace(output_path)
    Path(progress_path).unlink(missing_ok=True)
    return next_groups, time.process_time() - cpu_start, timing


def load_resume_state(output_dir, input_path, initial_groups, max_level, cardinality):
    level_zero = level_output_path(output_dir, input_path, 0)
    if not level_zero.exists():
        write_groups(initial_groups, level_zero)

    groups = read_groups(level_zero, cardinality)
    current_level = 0
    for level in range(1, max_level + 1):
        path = level_output_path(output_dir, input_path, level)
        if not path.exists():
            break
        try:
            groups = read_groups(path, cardinality)
        except ValueError:
            break
        current_level = level
    return current_level, groups


def ptr_embedding(args):
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()

    embedding_cpu_start = time.process_time()
    cardinality, num_tokens, dimensions = stream_representations(args.input, args.output)
    embedding_cpu_seconds = time.process_time() - embedding_cpu_start

    return {
        "meta": {
            "algorithm": "les3-ptr-embedding",
            "date": current_timestamp_json(),
            "cardinality": cardinality,
            "dataset": Path(args.input).name,
            "output": str(Path(args.output)),
        },
        "statistics": {
            "timing": {
                "embedding": embedding_cpu_seconds,
                "total_wall": time.perf_counter() - total_wall_start,
                "total_cpu": time.process_time() - total_cpu_start,
            },
            "embedding": {
                "num_tokens": num_tokens,
                "dimensions": dimensions,
            },
        },
    }


def train_l2p(args):
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()
    torch_threads = configure_torch_threads()

    cardinality, num_tokens, min_tokens, set_offsets = scan_set_file(args.input)
    partition_order = sorted(range(cardinality), key=lambda set_id: min_tokens[set_id])
    ptr_offsets, ptr_vector_dimensions = build_ptr_offsets(args.ptr, cardinality)

    initial_groups = initialize_groups(partition_order, INITIAL_GROUPS)
    levels, target_groups = deduce_level_count(cardinality, len(initial_groups))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_wall_start = time.perf_counter()
    partition_cpu_seconds = 0.0
    partition_timing = empty_partition_timing()
    resumed_from_level, groups = load_resume_state(output_dir, args.input, initial_groups, levels, cardinality)
    for level in range(resumed_from_level + 1, levels + 1):
        groups, level_cpu_seconds, level_timing = split_level_resumable(
            groups,
            args.input,
            args.ptr,
            set_offsets,
            ptr_offsets,
            ptr_vector_dimensions,
            output_dir,
            level,
            cardinality,
        )
        partition_cpu_seconds += level_cpu_seconds
        add_partition_timing(partition_timing, level_timing)
    partition_wall_seconds = time.perf_counter() - partition_wall_start

    final_output = final_output_path(output_dir, args.input)
    copy_final_level(level_output_path(output_dir, args.input, levels), final_output)

    meta = {
        "algorithm": "les3-precompute",
        "date": current_timestamp_json(),
        "cardinality": cardinality,
        "dataset": Path(args.input).name,
        "ptr_input": str(Path(args.ptr)),
        "output_dir": str(output_dir),
        "output": str(final_output),
        "levels": levels,
        "target_group_count": target_groups,
        "target_group_fraction": TARGET_GROUP_FRACTION,
        "initial_groups": INITIAL_GROUPS,
        "actual_initial_groups": len(initial_groups),
        "partition_order": "minimum_token",
        "resumed_from_level": resumed_from_level,
        "torch_threads": torch_threads,
    }
    label = getattr(args, "label", "")
    if label:
        meta["label"] = label

    return {
        "meta": meta,
        "statistics": {
            "timing": {
                "partition_wall": partition_wall_seconds,
                "partition_cpu": partition_cpu_seconds,
                **partition_timing,
            },
            "partitioning": {
                "initial_group_count": len(initial_groups),
                "final_group_count": len(groups),
                "target_group_count": target_groups,
                "num_tokens": num_tokens,
                "ptr_vector_dimensions": ptr_vector_dimensions,
            },
        },
    }
