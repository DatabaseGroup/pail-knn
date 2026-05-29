#ifndef KNN_PROJECT_JOIN_STATISTICS_HH
#define KNN_PROJECT_JOIN_STATISTICS_HH

#include "jemalloc_statistics.hh"

namespace statistics {

struct QueryStatistics {
  CountItem<> requested;
  CountItem<> performed;
  bool timed_out = false;

  void set(const int64_t requested_count, const int64_t performed_count, const bool did_time_out) {
    requested.value = requested_count;
    performed.value = performed_count;
    timed_out = did_time_out;
  }

  void add_to_json(const std::string& name, nlohmann::json& json) const {
    nlohmann::json queries;
    requested.add_to_json("requested", queries);
    performed.add_to_json("performed", queries);
    queries["timed_out"] = timed_out;
    json[name] = queries;
  }

  QueryStatistics& operator+=(const QueryStatistics& rhs) {
    requested.value += rhs.requested.value;
    performed.value += rhs.performed.value;
    timed_out = timed_out || rhs.timed_out;

    return *this;
  }
};

struct JoinStatistics {
  virtual ~JoinStatistics() = default;
  QueryStatistics queries;
  AvgItem<> heap_pops;
  AvgFloatItem<> final_k_similarity;
  CountItem<> precandidates;
  CountItem<> verifications;
  CountItem<> verified_tokens;
  AvgItem<> avg_verifications;
  CountItem<> successful_verifications;
  AvgItem<> final_heap_size;
  AvgItem<> batch_size;
  AvgItem<> heap_peak_bytes;

  [[nodiscard]] virtual nlohmann::json to_json() const {
    nlohmann::json json;

    heap_pops.add_to_json("heap_pops", json);
    precandidates.add_to_json("precandidates", json);
    verifications.add_to_json("verifications", json);
    verified_tokens.add_to_json("verified_tokens", json);
    successful_verifications.add_to_json("successful_verifications", json);
    avg_verifications.add_to_json("avg_verifications", json);
    final_k_similarity.add_to_json("final_k_similarity", json);
    final_heap_size.add_to_json("final_heap_size", json);
    batch_size.add_to_json("batch_size", json);
    heap_peak_bytes.add_to_json("heap_peak_bytes", json);
    queries.add_to_json("queries", json);

    return json;
  }

  virtual JoinStatistics& operator+=(const JoinStatistics& rhs) {
    queries += rhs.queries;
    precandidates.value += rhs.precandidates.value;
    verifications.value += rhs.verifications.value;
    verified_tokens.value += rhs.verified_tokens.value;
    successful_verifications.value += rhs.successful_verifications.value;

    combine_avg_item(heap_pops, rhs.heap_pops);
    combine_avg_item(avg_verifications, rhs.avg_verifications);
    combine_avg_item(final_heap_size, rhs.final_heap_size);
    combine_avg_item(final_k_similarity, rhs.final_k_similarity);
    combine_avg_item(batch_size, rhs.batch_size);
    combine_avg_item(heap_peak_bytes, rhs.heap_peak_bytes);

    return *this;
  }
};

}  // namespace statistics

#endif  // KNN_PROJECT_JOIN_STATISTICS_HH
