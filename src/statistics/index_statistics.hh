#ifndef KNN_PROJECT_INDEX_STATISTICS_HH
#define KNN_PROJECT_INDEX_STATISTICS_HH

#include <nlohmann/json.hpp>

#include "statistics.hh"

namespace statistics {

struct IndexStatistics {
  virtual ~IndexStatistics() = default;
  AvgItem<> list_size;
  AvgItem<> read_size;

  [[nodiscard]] virtual nlohmann::json to_json() const {
    nlohmann::json json;

    list_size.add_to_json("list_size", json);
    read_size.add_to_json("read_size", json);

    return json;
  }
};

struct PrefixStatistics : IndexStatistics {
  CountItem<> length_accesses;
  CountItem<> token_accesses;
  CountItem<> position_accesses;
  int64_t group_count{};
  HistItem<2> group_histogram;

  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json = IndexStatistics::to_json();

    length_accesses.add_to_json("length_accesses", json);
    token_accesses.add_to_json("token_accesses", json);
    position_accesses.add_to_json("position_accesses", json);
    json["group_count"] = group_count;
    group_histogram.add_to_json("group_histogram", json);

    return json;
  }
};

struct PartitionStatistics : IndexStatistics {
  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json = IndexStatistics::to_json();

    return json;
  }
};

}  // namespace statistics

#endif  // KNN_PROJECT_INDEX_STATISTICS_HH
