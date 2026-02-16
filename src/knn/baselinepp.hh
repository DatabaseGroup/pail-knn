#ifndef KNN_PROJECT_BASELINEPP_HH
#define KNN_PROJECT_BASELINEPP_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "../statistics/join_statistics.hh"
#include "topk_heap.hh"

namespace knn {

struct BaselinePPThreadState {
  statistics::JoinStatistics statistics;
};

template <class Similarity>
class BaselinePPJoin : public join::MultithreadedJoin<BaselinePPJoin<Similarity>, BaselinePPThreadState> {
public:
  using Base = join::MultithreadedJoin<BaselinePPJoin, BaselinePPThreadState>;

  struct active_query {
    size_t id;
    const Tokens* tokens;

    active_query(const size_t idx, const Tokens* t) : id(idx), tokens(t) {}
  };

public:
  BaselinePPJoin(data::SetData& data, const int32_t k, const int32_t min_batch_size, const int32_t max_batch_size)
      : Base(k, data), min_batch_size(min_batch_size), max_batch_size(max_batch_size) {
    int32_t current_size = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(Base::records.size()); ++i) {
      auto& record = Base::records[i];
      if (current_size != static_cast<int32_t>(record.tokens.size())) {
        current_size = static_cast<int32_t>(record.tokens.size());
        size_index.emplace_back(current_size, i);
      }
    }
  }

  // Single query version (unchanged for compatibility)
  TopKHeap linear_scan(int32_t record_id, BaselinePPThreadState& state) {
    auto& query = Base::records[record_id];
    TopKHeap ref(Base::k);
    auto r_size = static_cast<int32_t>(query.tokens.size());

    const auto it = std::ranges::lower_bound(size_index, std::make_pair(r_size, INT32_C(0)));
    const auto mid_idx = it->second;

    std::pair<int32_t, double> left, right;

    const auto left_idx = mid_idx - 1;
    update_similarity(query, left, left_idx);
    const auto right_idx = scan_group<false>(query, ref, mid_idx);
    update_similarity(query, right, right_idx);

    while (std::max(left.second, right.second) > ref.get_threshold()) {
      if (left.second > right.second) {
        const auto new_idx = scan_group<true>(query, ref, left.first);
        update_similarity(query, left, new_idx);
      } else {
        const auto new_idx = scan_group<false>(query, ref, right.first);
        update_similarity(query, right, new_idx);
      }
    }

    state.statistics.avg_verifications.record(right.first - left.first - 1);
    state.statistics.final_k_similarity.record(ref.get_threshold());
    state.statistics.precandidates.add(right.first - left.first - 1);
    state.statistics.verifications.add(right.first - left.first - 1);

    return ref;
  }

  void linear_scan_batch(const std::vector<int32_t>& query_ids,
                         BaselinePPThreadState& state,
                         std::vector<TopKHeap>& results) {
    const size_t batch_size = query_ids.size();
    results.clear();
    results.resize(batch_size, TopKHeap(Base::k));

    // all queries have the same size due to grouping
    const auto query_size = static_cast<int32_t>(Base::records[query_ids[0]].tokens.size());
    // track verifications, only add to all counters at the end
    std::vector<int64_t> verification_count(batch_size, 0);

    std::vector<active_query> active_queries;
    active_queries.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      active_queries.emplace_back(i, &Base::records[query_ids[i]].tokens);
    }

    // in this baseline, we assume that the size exists (always the case for self joins)
    const auto it = std::ranges::lower_bound(size_index, std::make_pair(query_size, INT32_C(0)));
    const int32_t mid_idx = it->second;
    int32_t left_idx = mid_idx - 1;
    int32_t right_idx = scan_group_batch<false>(active_queries, query_size, results, verification_count, mid_idx);

    double left_ub = compute_upper_bound(query_size, left_idx);
    double right_ub = compute_upper_bound(query_size, right_idx);

    while (!active_queries.empty() && (left_ub > 0 || right_ub > 0)) {
      // remove queries that can terminate
      double max_ub = std::max(left_ub, right_ub);
      remove_inactive_queries(active_queries, results, max_ub);

      if (active_queries.empty())
        break;

      if (left_ub > right_ub) {
        int32_t new_left_idx = scan_group_batch<true>(active_queries, query_size, results, verification_count, left_idx);
        left_idx = new_left_idx;
        left_ub = compute_upper_bound(query_size, left_idx);
      } else {
        int32_t new_right_idx = scan_group_batch<false>(active_queries, query_size, results, verification_count, right_idx);
        right_idx = new_right_idx;
        right_ub = compute_upper_bound(query_size, right_idx);
      }
    }

    // Record statistics
    state.statistics.batch_size.record(static_cast<int64_t>(batch_size));
    for (size_t i = 0; i < batch_size; ++i) {
      state.statistics.avg_verifications.record(verification_count[i]);
      state.statistics.final_k_similarity.record(results[i].get_threshold());
      state.statistics.precandidates.add(verification_count[i]);
      state.statistics.verifications.add(verification_count[i]);
    }
  }

  [[nodiscard]] nlohmann::json get_json_statistics() const {
    nlohmann::json json;
    nlohmann::json inner = Base::get_join_statistics();
    json["join"] = inner;

    json["timing"] = Base::timing.to_json();

    return json;
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    oneapi::tbb::task_arena arena(concurrency);
    oneapi::tbb::blocked_range<Iterator> iter(begin, end);

    auto cache_size = static_cast<double>(util::get_l2_cache_size());
    auto avg_set_size = Base::set_data.avg_set_size() * sizeof(Token);
    auto batch_size = static_cast<int32_t>(0.1 * (cache_size - avg_set_size) / avg_set_size);
    batch_size = std::max(min_batch_size, batch_size);
    batch_size = std::min(max_batch_size, batch_size);

    Base::timing.join_time.start();
    arena.execute([&] {
      oneapi::tbb::parallel_for(iter, [&](const tbb::blocked_range<Iterator>& r) {
        auto& state = Base::thread_states.local();

        // Group queries by size for batching
        HashTable<int32_t, std::vector<int32_t>> size_groups;

        for (auto it = r.begin(); it != r.end(); ++it) {
          int32_t record_id = *it;
          auto size = static_cast<int32_t>(Base::records[record_id].tokens.size());
          size_groups[size].push_back(record_id);
        }

        std::vector<TopKHeap> results;

        // Process each size group
        for (auto& group : size_groups | std::views::values) {
          // Process in batches
          for (int32_t start = 0; start < static_cast<int32_t>(group.size()); start += batch_size) {
            const auto end_idx = std::min(start + batch_size, static_cast<int32_t>(group.size()));
            std::vector batch(group.begin() + start, group.begin() + end_idx);

            linear_scan_batch(batch, state, results);
          }
        }
      });
    });
    Base::timing.join_time.stop();
  }

private:
  void remove_inactive_queries(std::vector<active_query>& active_queries,
                               const std::vector<TopKHeap>& results,
                               const double max_ub) {
    size_t i = 0;
    while (i < active_queries.size()) {
      if (size_t id = active_queries[i].id; results[id].get_threshold() >= max_ub) {
        // replace with last element and pop
        if (i != active_queries.size() - 1) {
          std::swap(active_queries[i], active_queries.back());
        }
        active_queries.pop_back();
      } else {
        ++i;
      }
    }
  }

  double compute_upper_bound(int32_t query_size, int32_t idx) {
    if (idx < 0 || idx >= static_cast<int32_t>(Base::records.size())) {
      return -1.0;
    }
    return Similarity::sim_ub(query_size, static_cast<int32_t>(Base::records[idx].tokens.size()), 0, 0);
  }

  template <bool LEFT>
  int32_t scan_group_batch(const std::vector<active_query>& active_queries,
                           int32_t query_size,
                           std::vector<TopKHeap>& results,
                           std::vector<int64_t>& verif_counts,
                           int32_t start_idx) {
    constexpr int32_t increment = LEFT ? -1 : 1;
    const auto num_records = static_cast<int32_t>(Base::records.size());

    const size_t current_size = Base::records[start_idx].tokens.size();

    int32_t i = start_idx;
    for (; ((LEFT && i >= 0) || (!LEFT && i < num_records)) && Base::records[i].tokens.size() == current_size;
         i += increment) {
      const auto& candidate = Base::records[i];
      const auto cand_size = static_cast<int32_t>(candidate.tokens.size());

      for (const auto& query : active_queries) {
        auto& query_tokens = *query.tokens;
        auto& heap = results[query.id];

        auto overlap_threshold = Similarity::eqo(query_size, cand_size, heap.get_threshold());
        int32_t result = util::falsify_or_compute(query_tokens, candidate.tokens, overlap_threshold);

        if (result > 0) {
          auto sim = Similarity::sim(query_size, cand_size, result);
          heap.push(i, sim);
        }
      }
    }

    int32_t processed = LEFT ? (start_idx - i) : (i - start_idx);
    for (const auto& query : active_queries) {
      verif_counts[query.id] += processed;
    }

    return i;
  }

  template <bool LEFT>
  int32_t scan_group(const Record& query, TopKHeap& ref, int32_t idx) {
    constexpr int32_t increment = LEFT ? -1 : 1;
    auto r_size = static_cast<int32_t>(query.tokens.size());

    auto current_size = Base::records[idx].tokens.size();

    int32_t i = idx;
    for (; ((LEFT && 0 <= i) || (!LEFT && i < static_cast<int32_t>(Base::records.size()))) &&
           Base::records[i].tokens.size() == current_size;
         i += increment) {
      auto& candidate = Base::records[i];

      auto overlap_threshold =
        Similarity::eqo(r_size, static_cast<int32_t>(candidate.tokens.size()), ref.get_threshold());
      int32_t result = util::falsify_or_compute(query.tokens, candidate.tokens, overlap_threshold);

      if (result > 0) {
        auto sim = Similarity::sim(r_size, static_cast<int32_t>(candidate.tokens.size()), result);
        ref.push(i, sim);
      }
    }

    return i;
  }

  void update_similarity(const Record& query, std::pair<int32_t, double>& e, int32_t idx) {
    if (0 <= idx && idx < static_cast<int32_t>(Base::records.size())) {
      e.first = idx;
      e.second = Similarity::sim_ub(
        static_cast<int32_t>(query.tokens.size()), static_cast<int32_t>(Base::records[idx].tokens.size()), 0, 0);
    } else {
      e.first = (idx < 0) ? -1 : static_cast<int32_t>(Base::records.size());
      e.second = -1.;
    }
  }

private:
  std::vector<std::pair<int32_t, int32_t>> size_index;
  const int32_t min_batch_size;
  const int32_t max_batch_size;
};
}  // namespace knn

#endif  // KNN_PROJECT_BASELINEPP_HH
