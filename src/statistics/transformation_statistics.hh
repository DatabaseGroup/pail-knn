#ifndef KNN_PROJECT_TRANSFORMATIONSTATISTICS_HH
#define KNN_PROJECT_TRANSFORMATIONSTATISTICS_HH

#include "join_statistics.hh"

namespace statistics {
struct TransformationStatistics : JoinStatistics {
  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json result = JoinStatistics::to_json();

    tree_accesses.add_to_json("tree_accesses", result);

    return result;
  }

  TransformationStatistics& operator+=(const JoinStatistics& rhs) override {
    JoinStatistics::operator+=(rhs);

    if (auto trhs = dynamic_cast<const TransformationStatistics*>(&rhs)) {
      tree_accesses.value += trhs->tree_accesses.value;
    }

    return *this;
  }

public:
  CountItem<> tree_accesses;
};
} // namespace statistics

namespace knn::transformation {
struct TransThreadState {
  statistics::TransformationStatistics statistics;
};

static tbb::enumerable_thread_specific<TransThreadState> TRANSFORMATION_STATISTICS;
}

#endif  // KNN_PROJECT_TRANSFORMATIONSTATISTICS_HH