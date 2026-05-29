#ifndef KNN_PROJECT_JOINTIMING_HH
#define KNN_PROJECT_JOINTIMING_HH

#include "timing.hh"

namespace timing {

class JoinTiming {
public:
  Timer preprocessing_time;
  Timer indexing_time;
  Timer join_time;

public:
  [[nodiscard]] nlohmann::json to_json() const {
    nlohmann::json json;

    preprocessing_time.add_to_json("preprocessing_time", json);
    indexing_time.add_to_json("indexing_time", json);
    join_time.add_to_json("join_time", json);

    return json;
  }
};

}  // namespace timing

#endif  // KNN_PROJECT_JOINTIMING_HH
