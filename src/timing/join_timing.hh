#ifndef KNN_PROJECT_JOINTIMING_HH
#define KNN_PROJECT_JOINTIMING_HH

#include "timing.hh"

namespace timing {

class JoinTiming {
public:
  Timer join_time;

public:
  [[nodiscard]] nlohmann::json to_json() const {
    nlohmann::json json;

    join_time.add_to_json("join_time", json);

    return json;
  }
};

}  // namespace timing

#endif  // KNN_PROJECT_JOINTIMING_HH
