#include <algorithm>
#include <limits>
#include <stdexcept>

#include "set_data.hh"

namespace data {
namespace {

constexpr size_t kMaxInitialHashReserve = 16'000'000;

struct DatasetStats {
  size_t total_tokens = 0;
  Token max_token = 0;
  bool has_tokens = false;
};

struct PreprocessResult {
  int32_t universe_size = 0;
  int32_t non_unique_universe = 0;
};

DatasetStats scan_dataset(const Records& records) {
  DatasetStats stats;

  for (const Record& record : records) {
    for (const Token token : record.tokens) {
      if (token < 0) {
        throw std::invalid_argument("preprocessing requires non-negative token IDs");
      }

      ++stats.total_tokens;
      if (!stats.has_tokens || token > stats.max_token) {
        stats.max_token = token;
        stats.has_tokens = true;
      }
    }
  }

  return stats;
}

bool should_use_dense_frequencies(const DatasetStats& stats, const PreprocessOptions& options) {
  if (!stats.has_tokens || options.dense_entries_per_token == 0) {
    return false;
  }

  const size_t dense_entries = static_cast<size_t>(stats.max_token) + 1;
  const size_t max_entries_for_density =
    stats.total_tokens > std::numeric_limits<size_t>::max() / options.dense_entries_per_token
      ? std::numeric_limits<size_t>::max()
      : stats.total_tokens * options.dense_entries_per_token;
  return dense_entries <= options.max_dense_entries && dense_entries <= max_entries_for_density;
}

PreprocessResult make_result(const size_t unique_token_count) {
  if (unique_token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error("too many unique tokens for int32_t metadata");
  }

  return {
    .universe_size = unique_token_count == 0 ? 0 : static_cast<int32_t>(unique_token_count - 1),
    .non_unique_universe = static_cast<int32_t>(unique_token_count),
  };
}

void relabel_with_dense_map(Records& records, const std::vector<Token>& relabeling) {
  for (Record& record : records) {
    for (Token& token : record.tokens) {
      token = relabeling[token];
    }
    std::ranges::sort(record.tokens);
  }
}

void relabel_with_hash_map(Records& records, const HashTable<Token, Token>& relabeling) {
  for (Record& record : records) {
    for (Token& token : record.tokens) {
      token = relabeling.find(token)->second;
    }
    std::ranges::sort(record.tokens);
  }
}

PreprocessResult preprocess_dense(Records& records, const DatasetStats& stats) {
  const size_t dense_entries = static_cast<size_t>(stats.max_token) + 1;
  std::vector<int32_t> frequencies(dense_entries, 0);
  std::vector<Token> used_tokens;
  used_tokens.reserve(std::min(stats.total_tokens, dense_entries));

  for (const Record& record : records) {
    for (const Token token : record.tokens) {
      if (frequencies[token] == 0) {
        used_tokens.push_back(token);
      }
      ++frequencies[token];
    }
  }

  std::ranges::sort(used_tokens, [&](const Token lhs, const Token rhs) { return frequencies[lhs] < frequencies[rhs]; });

  std::vector<Token> relabeling(dense_entries, -1);
  for (size_t i = 0; i < used_tokens.size(); ++i) {
    relabeling[used_tokens[i]] = static_cast<Token>(i);
  }

  relabel_with_dense_map(records, relabeling);

  return make_result(used_tokens.size());
}

PreprocessResult preprocess_hash(Records& records, const DatasetStats& stats) {
  HashTable<Token, int32_t> frequencies;
  frequencies.reserve(std::min(stats.total_tokens, kMaxInitialHashReserve));

  std::vector<Token> used_tokens;
  used_tokens.reserve(std::min(stats.total_tokens, kMaxInitialHashReserve));

  for (const Record& record : records) {
    for (const Token token : record.tokens) {
      auto [it, inserted] = frequencies.try_emplace(token, 0);
      if (inserted) {
        used_tokens.push_back(token);
      }
      ++it->second;
    }
  }

  std::ranges::sort(used_tokens, [&](const Token lhs, const Token rhs) {
    return frequencies.find(lhs)->second < frequencies.find(rhs)->second;
  });

  HashTable<Token, Token> relabeling;
  relabeling.reserve(used_tokens.size());
  for (size_t i = 0; i < used_tokens.size(); ++i) {
    relabeling.emplace(used_tokens[i], static_cast<Token>(i));
  }

  relabel_with_hash_map(records, relabeling);

  return make_result(used_tokens.size());
}

PreprocessResult preprocess_records(Records& records, const PreprocessOptions& options) {
  const DatasetStats stats = scan_dataset(records);

  if (!stats.has_tokens) {
    return make_result(0);
  }

  if (should_use_dense_frequencies(stats, options)) {
    return preprocess_dense(records, stats);
  }
  return preprocess_hash(records, stats);
}

}  // namespace

void SetData::preprocess(const PreprocessOptions& options) {
  const auto result = preprocess_records(records, options);

  std::ranges::sort(records,
                    [](const Record& lhs, const Record& rhs) { return lhs.tokens.size() < rhs.tokens.size(); });

  universe_size = result.universe_size;
  non_unique_universe = result.non_unique_universe;
}

}  // namespace data
