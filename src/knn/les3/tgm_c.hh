#ifndef KNN_PROJECT_LES3_TGM_C_HH
#define KNN_PROJECT_LES3_TGM_C_HH

#include <cstddef>
#include <roaring/roaring.hh>
#include <set>
#include <vector>

namespace knn::les3 {

using roaring::Roaring;
using std::multiset;
using std::vector;

struct TGM_C {
  vector<Roaring> bit_map;  // group-token

  void construct(const Records& records, const vector<vector<int32_t>>& groups);
  [[nodiscard]] double getUB(const std::vector<int32_t>& query_set, int32_t group_id) const;
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] int64_t get_size() const;
};

inline void TGM_C::construct(const Records& records, const vector<vector<int32_t>>& groups) {
  bit_map.clear();
  for (const auto& group : groups) {
    Roaring r;
    for (const int32_t set_id : group) {
      for (const int32_t token : records[set_id].tokens)
        r.add(token);
    }
    bit_map.push_back(r);
  }
}

inline double TGM_C::getUB(const std::vector<int32_t>& query_set, const int32_t group_id) const {
  // return Measure::computeUB(query_set, bit_map, group_id);
  int32_t num_common_tokens = 0;
  for (const int32_t token : query_set) {
    if (bit_map[group_id].contains(token))
      num_common_tokens += 1;
  }
  return static_cast<double>(num_common_tokens) / static_cast<double>(query_set.size());
}

inline std::size_t TGM_C::size_bytes() const {
  std::size_t total_size = 0;
  for (const auto& r : bit_map) {
    total_size += r.getSizeInBytes();
  }
  return total_size;
}

inline int64_t TGM_C::get_size() const {
  int64_t total_size = 0;
  for (const auto& r : bit_map) {
    total_size += static_cast<int64_t>(r.getSizeInBytes());
  }
  return total_size / 1024 / 1024;
}

}  // namespace knn::les3

#endif  // KNN_PROJECT_LES3_TGM_C_HH
