#ifndef KNN_PROJECT_LENGTH_GROUP_HH
#define KNN_PROJECT_LENGTH_GROUP_HH

namespace indexing {

using lengrp_idx = int32_t;

struct len_hist_entry {
  int32_t size;
  int32_t count;
  int64_t weight;

  explicit len_hist_entry(int32_t size) : size(size), count(0), weight(0) {}
};
struct len_hist {
  std::vector<len_hist_entry> entries;
  int64_t total_weight = 0;
  int32_t record_count = 0;
};

inline len_hist build_length_histogram(const Records& records) {
  int32_t current_size = -1;
  size_t i = 0;
  len_hist hist;

  while (i < records.size()) {
    current_size = static_cast<int32_t>(records[i].tokens.size());
    auto& current_group = hist.entries.emplace_back(current_size);

    while (i < records.size() && current_size == static_cast<int32_t>(records[i].tokens.size())) {
      current_group.weight += static_cast<int64_t>(records[i].tokens.size());
      current_group.count += 1;
      hist.total_weight += static_cast<int64_t>(records[i].tokens.size());
      ++i;
    }
  }
  hist.record_count = static_cast<int32_t>(records.size());
  return hist;
}

struct length_group {
  int32_t lower_bound;

  explicit length_group(int32_t lowerBound) : lower_bound(lowerBound) {}
};

class LengthGrouping {
public:
  explicit LengthGrouping(const len_hist& len_hist) : histogram(len_hist) {}

  static const bool LENGTH_IS_EXACT = false;

  [[nodiscard]] size_t size() const { return groups.size(); }

  // inclusive upper bound on set size
  [[nodiscard]] int32_t upper_bound(lengrp_idx i) const { return lower_bound(i + 1) - 1; }
  // inclusive lower bound on set size
  [[nodiscard]] int32_t lower_bound(lengrp_idx i) const { return groups[i].lower_bound; }

  [[nodiscard]] const std::vector<length_group>& get_groups() const { return groups; }

  void store_histogram(statistics::HistItem<2>& stat_hist) const {
    int32_t sidx = 0;
    for (auto& e : histogram.entries) {
      while (e.size > upper_bound(sidx)) {
        ++sidx;
      }
      stat_hist.record(e.size, {e.count, e.weight}, std::to_string(sidx));
    }
  }

protected:
  std::vector<length_group> groups;
  const len_hist& histogram;
};

class ExactLengthGrouping : public LengthGrouping {
public:
  static const bool LENGTH_IS_EXACT = true;

  explicit ExactLengthGrouping(const len_hist& len_hist) : LengthGrouping(len_hist) {
    for (auto& entry : len_hist.entries) {
      groups.emplace_back(entry.size);
    }

    // tombstone
    groups.emplace_back(len_hist.entries.back().size + 1);
  }
};

class BalancedFunctionalLengthGrouping : public LengthGrouping {
public:
  static const bool LENGTH_IS_EXACT = false;

  BalancedFunctionalLengthGrouping(const len_hist& len_hist, const std::function<size_t(size_t)>& group_count_fun)
      : LengthGrouping(len_hist) {
    auto group_count = static_cast<int32_t>(group_count_fun(len_hist.total_weight));

    auto is_possible = [&](const int64_t max_allowed_sum) -> bool {
      if (group_count == 0)
        return false;
      int32_t partitions_needed = 1;
      int64_t current_sum = 0;

      for (const auto& entry : len_hist.entries) {
        if (entry.weight > max_allowed_sum)
          return false;

        if (current_sum + entry.weight <= max_allowed_sum) {
          current_sum += entry.weight;
        } else {
          partitions_needed++;
          current_sum = entry.weight;
        }
      }
      return partitions_needed <= group_count;
    };

    const auto max_elem_it = std::ranges::max_element(len_hist.entries, {}, &len_hist_entry::weight);
    int64_t low = max_elem_it->weight;
    int64_t high = len_hist.total_weight;
    int64_t optimal_max_sum = high;

    // binary search on best split point
    while (low <= high) {
      if (const int64_t mid = std::midpoint(low, high); is_possible(mid)) {
        optimal_max_sum = mid;
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }

    size_t i = 0;
    while (i < len_hist.entries.size()) {
      int32_t size_start = len_hist.entries[i].size;
      groups.emplace_back(size_start);
      int64_t group_weight = 0;
      while (i < len_hist.entries.size() && group_weight + len_hist.entries[i].weight <= optimal_max_sum) {
        group_weight += len_hist.entries[i].weight;
        ++i;
      }
    }

    // tombstone
    groups.emplace_back(len_hist.entries.back().size + 1);
  }
};

class AIOLengthGrouping : public LengthGrouping {
public:
  static const bool LENGTH_IS_EXACT = false;

  explicit AIOLengthGrouping(const len_hist& len_hist) : LengthGrouping(len_hist) {
    groups.emplace_back(1);

    // tombstone
    groups.emplace_back(len_hist.entries.back().size + 1);
  }
};

class FixedWidthLengthGrouping : public LengthGrouping {
public:
  static const bool LENGTH_IS_EXACT = false;

  explicit FixedWidthLengthGrouping(const len_hist& len_hist, int32_t width) : LengthGrouping(len_hist) {
    int32_t s = len_hist.entries.front().size;
    while (s < len_hist.entries.back().size) {
      groups.emplace_back(s);
      s += width;
    }

    // tombstone
    groups.emplace_back(len_hist.entries.back().size + 1);
  }
};

}  // namespace indexing

#endif  // KNN_PROJECT_LENGTH_GROUP_HH
