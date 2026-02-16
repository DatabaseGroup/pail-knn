#ifndef KNN_PROJECT_TIMING_HH
#define KNN_PROJECT_TIMING_HH

namespace timing {

using clock = std::chrono::steady_clock;
using time_point = std::chrono::time_point<clock>;
using time_duration = std::chrono::duration<double>;

struct DummyTimer {
  void start() {}
  void stop() {}
  void clear() {}
  void add_to_json([[maybe_unused]] const std::string& name, [[maybe_unused]] nlohmann::json& json) const {}

  [[nodiscard]] double get() const { return time_duration::zero().count(); }
};

struct RealTimer {
  void start() { begin = clock::now(); }
  void stop() { duration += clock::now() - begin; }
  void clear() { duration = time_duration::zero(); }
  [[nodiscard]] double get() const { return duration.count(); }
  void add_to_json(const std::string& name, nlohmann::json& json) const { json[name] = get(); }

public:
  time_duration duration = time_duration::zero();
  time_point begin;
};

#ifdef TIMING
using Timer = RealTimer;
#else
using Timer = DummyTimer;
#endif

}  // namespace timing

#endif  // KNN_PROJECT_TIMING_HH
