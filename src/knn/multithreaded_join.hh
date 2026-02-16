#ifndef KNN_PROJECT_MULTITHREADED_JOIN_HH
#define KNN_PROJECT_MULTITHREADED_JOIN_HH

#include <oneapi/tbb/enumerable_thread_specific.h>

#include <boost/range/irange.hpp>
#include <nlohmann/json.hpp>

#include "../statistics/join_statistics.hh"
#include "../timing/join_timing.hh"

namespace join {

// Using Curiously Recurring Template Pattern
template <class Derived, class ThreadState>
class MultithreadedJoin {
public:
  MultithreadedJoin(const int32_t k, const data::SetData& set_data) : k(k), set_data(set_data), records(set_data.get_records()) {}

  void join(const int32_t concurrency = -1) {
    auto all_records = boost::irange<int32_t>(0, static_cast<int32_t>(records.size()));
    static_cast<Derived*>(this)->do_multiple_lookups(all_records.begin(), all_records.end(), concurrency);
  }

  void sample_and_query(const int32_t sample_size,
                        const uint64_t seed = std::random_device()(),
                        const int32_t concurrency = -1) {
    std::mt19937 rand(seed);
    std::uniform_int_distribution dist(0, static_cast<int32_t>(records.size() - 1));

    RecordIds sample;
    sample.reserve(sample_size);
    for (int32_t i = 0; i < sample_size; ++i) {
      sample.push_back(dist(rand));
    }
    std::ranges::sort(sample);

    static_cast<Derived*>(this)->do_multiple_lookups(sample.begin(), sample.end(), concurrency);
  }

protected:
  nlohmann::json get_join_statistics() const {
    statistics::JoinStatistics final_result;
    for (auto& s : thread_states) {
      final_result += s.statistics;
    }
    return final_result.to_json();
  }

protected:
  int32_t k;
  const data::SetData& set_data;
  const Records& records;
  tbb::enumerable_thread_specific<ThreadState> thread_states;
  timing::JoinTiming timing{};
};

}  // namespace join

#endif  // KNN_PROJECT_MULTITHREADED_JOIN_HH
