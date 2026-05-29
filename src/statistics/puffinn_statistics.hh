#ifndef KNN_PROJECT_PUFFINN_STATISTICS_HH
#define KNN_PROJECT_PUFFINN_STATISTICS_HH

#include "join_statistics.hh"

namespace statistics {

struct PuffinnJoinStatistics : JoinStatistics {
  AvgItem<> considered_maps{};
  AvgItem<> hash_length{};

  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json result = JoinStatistics::to_json();

    considered_maps.add_to_json("considered_maps", result);
    hash_length.add_to_json("hash_length", result);

    return result;
  }

  PuffinnJoinStatistics& operator+=(const JoinStatistics& rhs) override {
    JoinStatistics::operator+=(rhs);

    if (auto prhs = dynamic_cast<const PuffinnJoinStatistics*>(&rhs)) {
      combine_avg_item(considered_maps, prhs->considered_maps);
      combine_avg_item(hash_length, prhs->hash_length);
    }

    return *this;
  }
};

};

#endif  // KNN_PROJECT_PUFFINN_STATISTICS_HH
