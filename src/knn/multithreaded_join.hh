#ifndef KNN_PROJECT_MULTITHREADED_JOIN_HH
#define KNN_PROJECT_MULTITHREADED_JOIN_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <atomic>
#include <boost/range/irange.hpp>
#include <chrono>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <utility>

#include "../statistics/index_statistics.hh"
#include "../statistics/join_statistics.hh"
#include "../timing/join_timing.hh"

namespace join {

// Using Curiously Recurring Template Pattern
template <class Derived, class ThreadState>
class MultithreadedJoin {
public:
  static constexpr int64_t TIMEOUT_CHECK_INTERVAL = 1000;

  MultithreadedJoin(const int32_t k, const data::SetData& set_data)
      : k(k), set_data(set_data), records(set_data.get_records()) {}

  void set_setup_timing(const timing::JoinTiming& setup_timing) { timing = setup_timing; }

  void join(const int32_t concurrency = -1, const int64_t timeout_seconds = 0) {
    auto all_records = boost::irange<int32_t>(0, static_cast<int32_t>(records.size()));
    start_query_run(static_cast<int64_t>(records.size()), timeout_seconds);
    static_cast<Derived*>(this)->do_multiple_lookups(all_records.begin(), all_records.end(), concurrency);
  }

  void sample_and_query(const int32_t sample_size,
                        const uint64_t seed = std::random_device()(),
                        const int32_t concurrency = -1,
                        const int64_t timeout_seconds = 0) {
    std::mt19937 rand(seed);
    std::uniform_int_distribution dist(0, static_cast<int32_t>(records.size() - 1));

    RecordIds sample;
    sample.reserve(sample_size);
    for (int32_t i = 0; i < sample_size; ++i) {
      sample.push_back(dist(rand));
    }
    std::ranges::sort(sample);

    start_query_run(sample_size, timeout_seconds);
    static_cast<Derived*>(this)->do_multiple_lookups(sample.begin(), sample.end(), concurrency);
  }

protected:
  template <class Iterator, class SetupFn, class LookupFn, class CleanupFn>
  void run_parallel_lookups(Iterator begin,
                            Iterator end,
                            const int32_t concurrency,
                            SetupFn setup,
                            LookupFn lookup,
                            CleanupFn cleanup) {
    oneapi::tbb::task_arena arena(concurrency);
    oneapi::tbb::blocked_range<Iterator> iter(begin, end);

    sample_jemalloc_heap();
    timing.join_time.start();
    arena.execute([&] {
      oneapi::tbb::auto_partitioner partitioner;
      oneapi::tbb::parallel_for(
        iter,
        [&](const oneapi::tbb::blocked_range<Iterator>& r) {
          auto& state = thread_states.local();
          setup(state);
          sample_jemalloc_heap(state.statistics);
          for (auto it = r.begin(); it != r.end(); ++it) {
            if (!should_start_query()) {
              break;
            }
            const auto record_id = *it;
            lookup(record_id, state);
            cleanup(record_id, state);
            record_completed_queries();
          }
          sample_jemalloc_heap(state.statistics);
        },
        partitioner);
    });
    timing.join_time.stop();
    sample_jemalloc_heap();
  }

  void sample_jemalloc_heap() { sample_jemalloc_heap(thread_states.local().statistics); }

  static void sample_jemalloc_heap(statistics::JoinStatistics& join_statistics) {
    statistics::record_jemalloc_heap_bytes(join_statistics.heap_peak_bytes);
  }

  nlohmann::json get_join_statistics() const {
    using Statistics = std::remove_cvref_t<decltype(std::declval<ThreadState&>().statistics)>;

    Statistics final_result{};
    for (auto& s : thread_states) {
      final_result += s.statistics;
    }
    final_result.queries = get_query_statistics();
    return final_result.to_json();
  }

  [[nodiscard]] statistics::QueryStatistics get_query_statistics() const {
    const auto performed = performed_queries.load(std::memory_order_relaxed);
    statistics::QueryStatistics query_statistics;
    query_statistics.set(requested_queries,
                         performed,
                         timed_out.load(std::memory_order_relaxed) && performed < requested_queries);
    return query_statistics;
  }

  template <class IndexStatistics>
  IndexStatistics get_index_statistics(const IndexStatistics& base_statistics,
                                       IndexStatistics ThreadState::* thread_statistics) const {
    IndexStatistics final_result = base_statistics;
    for (auto& s : thread_states) {
      final_result += s.*thread_statistics;
    }
    return final_result;
  }

  void record_completed_queries(const int64_t count = 1) {
    performed_queries.fetch_add(count, std::memory_order_relaxed);
    if (timeout_seconds <= 0 || timed_out.load(std::memory_order_relaxed)) {
      return;
    }

    const auto previous = timeout_check_counter.fetch_add(count, std::memory_order_relaxed);
    const auto current = previous + count;
    if (previous / TIMEOUT_CHECK_INTERVAL != current / TIMEOUT_CHECK_INTERVAL) {
      check_timeout();
    }
  }

  [[nodiscard]] bool should_start_query() const { return !stop_queries.load(std::memory_order_acquire); }

protected:
  int32_t k;
  const data::SetData& set_data;
  const Records& records;
  tbb::enumerable_thread_specific<ThreadState> thread_states;
  timing::JoinTiming timing{};

private:
  void start_query_run(const int64_t requested, const int64_t timeout) {
    requested_queries = requested;
    timeout_seconds = timeout;
    performed_queries.store(0, std::memory_order_relaxed);
    timeout_check_counter.store(0, std::memory_order_relaxed);
    timed_out.store(false, std::memory_order_relaxed);
    stop_queries.store(false, std::memory_order_release);

    if (timeout_seconds > 0) {
      query_deadline = timing::clock::now() + std::chrono::seconds(timeout_seconds);
    }
  }

  void check_timeout() {
    if (timeout_seconds <= 0 || timing::clock::now() < query_deadline) {
      return;
    }
    timed_out.store(true, std::memory_order_relaxed);
    stop_queries.store(true, std::memory_order_release);
  }

  int64_t requested_queries = 0;
  int64_t timeout_seconds = 0;
  std::atomic<int64_t> performed_queries{0};
  std::atomic<int64_t> timeout_check_counter{0};
  std::atomic<bool> timed_out{false};
  std::atomic<bool> stop_queries{false};
  timing::time_point query_deadline{};
};

}  // namespace join

#endif  // KNN_PROJECT_MULTITHREADED_JOIN_HH
