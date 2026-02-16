#ifndef KNN_PROJECT_PERF_HH
#define KNN_PROJECT_PERF_HH

#include <chrono>

#include "absl/strings/str_format.h"
#include "statistics.hh"

namespace statistics {
#ifndef STATISTICS

struct JoinPerf {
public:
  CountItem<> queries;
  CountItem<> heap_pops;
  CountItem<> precandidates;
  CountItem<> verifications;
  CountItem<> successful_verifications;
  AvgItem<> probe_set_size;
  AvgFloatItem<> final_k_similarity;
  AvgFloatItem<> ratio_dataset_read;

  void print_and_reset([[maybe_unused]] std::ostream& os) {
  }

  void reset() {
  }
};

#else

struct JoinPerf {
public:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;

  CountItem<> queries;
  CountItem<> heap_pops;
  CountItem<> precandidates;
  CountItem<> verifications;
  CountItem<> successful_verifications;
  AvgItem<> probe_set_size;
  AvgFloatItem<> final_k_similarity;
  AvgFloatItem<> ratio_dataset_read;

  void print_and_reset(std::ostream& os) {
    time_point current = clock::now();
    std::chrono::duration<double> diff = current - last_time;

    os << absl::StrFormat("queries             : %f/s\n", static_cast<double>(queries.value) / diff.count());
    os << absl::StrFormat("heap_pops           : %f/s\n", static_cast<double>(heap_pops.value) / diff.count());
    os << absl::StrFormat("precandidates       : %f/s\n", static_cast<double>(precandidates.value) / diff.count());
    os << absl::StrFormat("verifications       : %f/s\n", static_cast<double>(verifications.value) / diff.count());
    os << absl::StrFormat("succ. verifications : %f/s\n",
                          static_cast<double>(successful_verifications.value) / diff.count());
    os << absl::StrFormat("probe_set_size      : %f\n", probe_set_size.avg());
    os << absl::StrFormat("final_k_similarity  : %f\n", final_k_similarity.avg());
    os << absl::StrFormat("ratio_dataset_read  : %f\n", ratio_dataset_read.avg());
    os << std::endl;

    reset();
  }

  void reset() {
    queries.reset();
    heap_pops.reset();
    precandidates.reset();
    verifications.reset();
    successful_verifications.reset();
    probe_set_size.reset();
    final_k_similarity.reset();
    ratio_dataset_read.reset();
    last_time = clock::now();
  }

private:
  time_point last_time;
};

#endif
} // namespace statistics

#endif  // KNN_PROJECT_PERF_HH