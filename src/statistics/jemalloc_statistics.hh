#ifndef KNN_PROJECT_JEMALLOC_STATISTICS_HH
#define KNN_PROJECT_JEMALLOC_STATISTICS_HH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef USE_JEMALLOC
// keep je_* (otherwise unavailable on Linux)
#define JEMALLOC_NO_DEMANGLE
#include <jemalloc/jemalloc.h>
#endif

#include "statistics.hh"

namespace statistics {

inline void record_jemalloc_heap_bytes(AvgItem<>& item) {
#ifdef USE_JEMALLOC
  std::uint64_t epoch = 1;
  std::size_t epoch_size = sizeof(epoch);
  je_mallctl("epoch", &epoch, &epoch_size, &epoch, sizeof(epoch));

  std::size_t allocated = 0;
  std::size_t allocated_size = sizeof(allocated);
  const int result = je_mallctl("stats.allocated", &allocated, &allocated_size, nullptr, 0);
  if (result == 0) {
    constexpr auto max_value = static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
    item.record(static_cast<int64_t>(std::min(allocated, max_value)));
  }
#else
  (void)item;
#endif
}

}  // namespace statistics

#endif  // KNN_PROJECT_JEMALLOC_STATISTICS_HH
