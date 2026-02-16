#ifndef KNN_PROJECT_CACHE_HH
#define KNN_PROJECT_CACHE_HH

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>

namespace util {
inline size_t get_l2_cache_size() {
  size_t size = 0;
  size_t len = sizeof(size);

  if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0 &&
      len == sizeof(size) && size > 0) {
    return size;
  }

  // fallback
  return 256 * 1024;
}
}

#elif defined(__linux__)
#include <fstream>
#include <string>

namespace util {
inline size_t get_l2_cache_size() {
  const char* path = "/sys/devices/system/cpu/cpu0/cache/index2/size";

  std::ifstream file(path);
  if (!file.is_open()) {
    // fallback
    return 256 * 1024;
  }

  std::string value;
  file >> value;

  if (value.empty()) {
    // fallback
    return 256 * 1024;
  }

  // Expected formats: "256K", "1024K", "1M"
  size_t size = std::stoul(value);
  char suffix = value.back();

  if (suffix == 'K' || suffix == 'k') {
    size *= 1024;
  } else if (suffix == 'M' || suffix == 'm') {
    size *= 1024 * 1024;
  }

  return size > 0 ? size : 256 * 1024;
}
}

#else

namespace util {
inline size_t get_l2_cache_size() {
  // fallback
  return 256 * 1024;
}
}

#endif

#endif //KNN_PROJECT_CACHE_HH