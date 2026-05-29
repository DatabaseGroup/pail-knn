#ifndef KNN_PROJECT_BASELINE_HH
#define KNN_PROJECT_BASELINE_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "../statistics/join_statistics.hh"
#include "../util/cache.hh"
#include "../util/set_ops.hh"
#include "multithreaded_join.hh"
#include "topk_heap.hh"

namespace knn {

struct BaselineThreadState {
  statistics::JoinStatistics statistics;
};

template <class Similarity>
class BaselineJoin : public join::MultithreadedJoin<BaselineJoin<Similarity>, BaselineThreadState> {
public:
  using Base = join::MultithreadedJoin<BaselineJoin, BaselineThreadState>;

public:
  BaselineJoin(data::SetData& data, const int32_t k, const int32_t min_batch_size, const int32_t max_batch_size)
      : Base(k, data), min_batch_size(min_batch_size), max_batch_size(max_batch_size) {}

  void linear_scan_batch(const std::vector<int32_t>& query_ids,
                         BaselineThreadState& state,
                         std::vector<TopKHeap>& results) {
    const size_t batch_size = query_ids.size();
    results.clear();
    results.resize(query_ids.size(), TopKHeap(Base::k));

    std::vector<const Tokens*> query_records;
    query_records.reserve(query_ids.size());
    for (auto id : query_ids) {
      query_records.push_back(&Base::records[id].tokens);
    }

    for (int32_t cand_id = 0; cand_id < static_cast<int32_t>(Base::records.size()); ++cand_id) {
      const auto& s = Base::records[cand_id];
      const auto s_size = static_cast<int32_t>(s.tokens.size());

      for (size_t i = 0; i < batch_size; ++i) {
        auto& query_tokens = *query_records[i];
        auto& heap = results[i];

        auto overlap_threshold = Similarity::eqo(query_tokens.size(), s_size, heap.get_threshold());

        int32_t result = util::falsify_or_compute(query_tokens, s.tokens, overlap_threshold);

        if (result > 0) {
          auto sim = Similarity::sim(query_tokens.size(), s_size, result);
          heap.push(cand_id, sim);
        }
      }
    }

    state.statistics.batch_size.record(static_cast<int64_t>(batch_size));
    state.statistics.precandidates.add(static_cast<int64_t>(Base::records.size()) * batch_size);
    state.statistics.verifications.add(static_cast<int64_t>(Base::records.size()) * batch_size);
    for (auto& h : results) {
      state.statistics.avg_verifications.record(static_cast<int64_t>(Base::records.size()));
      state.statistics.final_k_similarity.record(h.get_threshold());
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
    auto cache_size = static_cast<double>(util::get_l2_cache_size());
    auto avg_set_size = Base::set_data.avg_set_size() * sizeof(Token);
    auto batch_size = static_cast<int32_t>(0.1 * (cache_size - avg_set_size) / avg_set_size);
    batch_size = std::max(min_batch_size, batch_size);
    batch_size = std::min(max_batch_size, batch_size);

    oneapi::tbb::task_arena arena(concurrency);
    oneapi::tbb::blocked_range<Iterator> iter(begin, end, batch_size);

    Base::sample_jemalloc_heap();
    Base::timing.join_time.start();
    arena.execute([&] {
      oneapi::tbb::parallel_for(iter, [&](const tbb::blocked_range<Iterator>& r) {
        auto& state = Base::thread_states.local();
        std::vector<int32_t> batch;
        batch.reserve(batch_size);
        std::vector<TopKHeap> results;
        Base::sample_jemalloc_heap(state.statistics);

        for (auto it = r.begin(); it != r.end(); ++it) {
          if (!Base::should_start_query()) {
            break;
          }
          batch.push_back(*it);

          if (static_cast<int32_t>(batch.size()) == batch_size) {
            linear_scan_batch(batch, state, results);
            Base::record_completed_queries(static_cast<int64_t>(batch.size()));
            batch.clear();
          }
        }

        // handle remaining queries
        if (!batch.empty() && Base::should_start_query()) {
          linear_scan_batch(batch, state, results);
          Base::record_completed_queries(static_cast<int64_t>(batch.size()));
        }
        Base::sample_jemalloc_heap(state.statistics);
      });
    });
    Base::timing.join_time.stop();
    Base::sample_jemalloc_heap();
  }

private:
  const int32_t min_batch_size;
  const int32_t max_batch_size;
};
}  // namespace knn

#endif  // KNN_PROJECT_BASELINE_HH
