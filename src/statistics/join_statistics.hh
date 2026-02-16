#ifndef KNN_PROJECT_JOIN_STATISTICS_HH
#define KNN_PROJECT_JOIN_STATISTICS_HH

namespace statistics {

struct JoinStatistics {
  virtual ~JoinStatistics() = default;
  AvgItem<> heap_pops;
  AvgFloatItem<> final_k_similarity;
  CountItem<> precandidates;
  CountItem<> verifications;
  CountItem<> verified_tokens;
  AvgItem<> avg_verifications;
  CountItem<> successful_verifications;
  AvgItem<> final_heap_size;
  AvgItem<> batch_size;

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

    return json;
  }

  virtual JoinStatistics& operator+=(const JoinStatistics& rhs) {
    precandidates.value += rhs.precandidates.value;
    verifications.value += rhs.verifications.value;
    verified_tokens.value += rhs.verified_tokens.value;
    successful_verifications.value += rhs.successful_verifications.value;

    auto combine_avg = [](auto& lhs, const auto& rhs_item) {
      lhs.sum += rhs_item.sum;
      lhs.count += rhs_item.count;
      lhs.min = std::min(lhs.min, rhs_item.min);
      lhs.max = std::max(lhs.max, rhs_item.max);
    };

    combine_avg(heap_pops, rhs.heap_pops);
    combine_avg(avg_verifications, rhs.avg_verifications);
    combine_avg(final_heap_size, rhs.final_heap_size);
    combine_avg(final_k_similarity, rhs.final_k_similarity);
    combine_avg(batch_size, rhs.batch_size);

    return *this;
  }
};


}  // namespace statistics

#endif  // KNN_PROJECT_JOIN_STATISTICS_HH
