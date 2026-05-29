#ifndef KNN_PROJECT_PUFFINN_HH
#define KNN_PROJECT_PUFFINN_HH

#include <nlohmann/json.hpp>
#include <ostream>
#include <puffinn/collection.hpp>

// This include here is required for some reason, puffin forgot to add the inclusion in their header file
#include <puffinn/math.hpp>
#include <puffinn/performance.hpp>
#include <puffinn/similarity_measure/jaccard.hpp>
#include <stdexcept>
#include <streambuf>
#include <vector>

#include "../similarity/Similarity.hh"
#include "../statistics/join_statistics.hh"
#include "../statistics/puffinn_statistics.hh"
#include "../util/set_ops.hh"
#include "multithreaded_join.hh"
#include "topk_heap.hh"

namespace knn {

class PuffinnCountingStreambuf : public std::streambuf {
public:
  [[nodiscard]] uint64_t bytes_written() const { return bytes_written_; }

protected:
  std::streamsize xsputn(const char*, const std::streamsize count) override {
    bytes_written_ += static_cast<uint64_t>(count);
    return count;
  }

  int_type overflow(const int_type ch) override {
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
      ++bytes_written_;
    }
    return traits_type::not_eof(ch);
  }

private:
  uint64_t bytes_written_{};
};

struct PuffinnThreadState {
  statistics::PuffinnJoinStatistics statistics;
};

struct PuffinnIndexStatistics : statistics::SizeOnlyIndexStatistics {
  uint64_t memory_budget_bytes{};
  float recall{};
  size_t repetitions{};
  uint32_t indexed_size{};
  uint32_t universe_size{};

  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json = statistics::SizeOnlyIndexStatistics::to_json();

    json["memory_budget_bytes"] = memory_budget_bytes;
    json["recall"] = recall;
    json["repetitions"] = repetitions;
    json["table_count"] = repetitions;
    json["indexed_size"] = indexed_size;
    json["universe_size"] = universe_size;

    return json;
  }
};

class PuffinnJoin : public join::MultithreadedJoin<PuffinnJoin, PuffinnThreadState> {
public:
  using Base = join::MultithreadedJoin<PuffinnJoin, PuffinnThreadState>;

  PuffinnJoin(data::SetData& data, const int32_t k, const uint64_t memory_budget_bytes, const float recall)
      : Base(k, data),
        index(static_cast<unsigned int>(data.get_universe_size() + 1), memory_budget_bytes),
        recall(recall) {
    const auto puffinn_universe_size = static_cast<uint32_t>(data.get_universe_size() + 1);

    puffinn_records.reserve(Base::records.size());
    for (const auto& record : Base::records) {
      puffinn_records.push_back(to_puffinn_tokens(record.tokens));
      index.insert(puffinn_records.back());
    }
    index.rebuild();

    index_statistics.set_size_bytes(memory_budget_bytes);
    index_statistics.memory_budget_bytes = memory_budget_bytes;
    index_statistics.recall = recall;
    index_statistics.repetitions = index.get_repetitions();
    index_statistics.indexed_size = index.get_size();
    index_statistics.universe_size = puffinn_universe_size;
  }

  [[nodiscard]] nlohmann::json get_json_statistics() const {
    nlohmann::json json;
    auto index_json = index_statistics.to_json();
    index_json["size_bytes"] = serialized_index_size_bytes();

    json["join"] = Base::get_join_statistics();
    json["index"] = index_json;
    json["timing"] = Base::timing.to_json();

    return json;
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    if (concurrency != 1) {
      throw std::invalid_argument("puffinn requires -c 1 because Index::search is not thread-safe");
    }

    auto& state = Base::thread_states.local();
    puffinn::g_performance_metrics.clear();

    Base::sample_jemalloc_heap();
    Base::timing.join_time.start();
    Base::sample_jemalloc_heap(state.statistics);
    size_t lookup_count = 0;
    for (auto it = begin; it != end; ++it) {
      if (!Base::should_start_query()) {
        break;
      }
      lookup(*it, state);
      ++lookup_count;
      Base::record_completed_queries();
    }
    Base::sample_jemalloc_heap(state.statistics);
    Base::timing.join_time.stop();
    aggregate_puffinn_metrics(state, lookup_count);
    Base::sample_jemalloc_heap();
  }

private:
  void aggregate_puffinn_metrics(PuffinnThreadState& state, const size_t lookup_count) {
    const auto metrics = puffinn::g_performance_metrics.get_query_metrics();

    // only read the last lookup_count metrics
    const auto skip_count = metrics.size() > lookup_count ? metrics.size() - lookup_count : 0;

    for (size_t i = skip_count; i < metrics.size(); ++i) {
      const auto& metric = metrics[i];
      state.statistics.hash_length.record(metric.hash_length);
      state.statistics.considered_maps.record(metric.considered_maps);
      state.statistics.precandidates.add(static_cast<int64_t>(metric.candidates));
      state.statistics.verifications.add(static_cast<int64_t>(metric.distance_computations));
      state.statistics.avg_verifications.record(static_cast<int64_t>(metric.distance_computations));
    }
  }

  void lookup(const int32_t record_id, PuffinnThreadState& state) {
    const auto& query_tokens = puffinn_records[record_id];
    const auto result_ids = index.search(query_tokens, static_cast<unsigned int>(Base::k), recall);

    // slightly redundant work, but not noticeable overall
    TopKHeap heap(Base::k);
    for (auto result_id : result_ids) {
      const auto candidate_id = static_cast<int32_t>(result_id);
      const auto& query = Base::records[record_id];
      const auto& candidate = Base::records[candidate_id];
      const auto overlap = util::exact_overlap(query.tokens, candidate.tokens);
      const auto similarity = similarity::JaccardSimilarity::sim(
        static_cast<int32_t>(query.tokens.size()), static_cast<int32_t>(candidate.tokens.size()), overlap);
      heap.push(candidate_id, similarity);
    }

    state.statistics.heap_pops.record(0);
    state.statistics.final_k_similarity.record(heap.get_threshold());
    state.statistics.final_heap_size.record(0);
  }

  static std::vector<uint32_t> to_puffinn_tokens(const Tokens& tokens) {
    std::vector<uint32_t> result;
    result.reserve(tokens.size());
    for (const auto token : tokens) {
      result.push_back(static_cast<uint32_t>(token));
    }
    return result;
  }

  [[nodiscard]] uint64_t serialized_index_size_bytes() const {
    PuffinnCountingStreambuf buffer;
    std::ostream out(&buffer);
    index.serialize(out);
    return buffer.bytes_written();
  }

private:
  puffinn::Index<puffinn::JaccardSimilarity> index;
  std::vector<std::vector<uint32_t>> puffinn_records;
  float recall;
  PuffinnIndexStatistics index_statistics;
};

}  // namespace knn

#endif  // KNN_PROJECT_PUFFINN_HH
