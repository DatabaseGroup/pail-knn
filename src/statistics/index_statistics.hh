#ifndef KNN_PROJECT_INDEX_STATISTICS_HH
#define KNN_PROJECT_INDEX_STATISTICS_HH

#include <cstddef>
#include <nlohmann/json.hpp>
#include <vector>

#include "statistics.hh"

namespace statistics {

template <class T, class Allocator>
[[nodiscard]] std::size_t vector_payload_bytes(const std::vector<T, Allocator>& v) {
  return v.capacity() * sizeof(T);
}

template <class T, class Allocator>
[[nodiscard]] std::size_t vector_allocated_bytes(const std::vector<T, Allocator>& v) {
  return sizeof(v) + vector_payload_bytes(v);
}

template <class HashMap>
[[nodiscard]] std::size_t flat_hash_map_payload_bytes(const HashMap& map) {
  // The container uses O((sizeof(std::pair<const K, V>) + 1) * bucket_count()) bytes.
  // https://abseil.io/docs/cpp/guides/container
  return map.bucket_count() * sizeof(std::pair<typename HashMap::key_type, typename HashMap::value_type>) + 1;
}

template <class HashMap>
[[nodiscard]] std::size_t flat_hash_map_allocated_bytes(const HashMap& map) {
  return sizeof(map) + flat_hash_map_payload_bytes(map);
}

struct IndexStatistics {
  virtual ~IndexStatistics() = default;
  int64_t size_bytes = 0;
  AvgItem<> list_size;
  AvgItem<> read_size;

  void set_size_bytes(const std::size_t bytes) { size_bytes = static_cast<int64_t>(bytes); }

  virtual IndexStatistics& operator+=(const IndexStatistics& rhs) {
    combine_avg_item(read_size, rhs.read_size);

    return *this;
  }

  [[nodiscard]] virtual nlohmann::json to_json() const {
    nlohmann::json json;

    json["size_bytes"] = size_bytes;
    list_size.add_to_json("list_size", json);
    read_size.add_to_json("read_size", json);

    return json;
  }
};

struct SizeOnlyIndexStatistics : IndexStatistics {
  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json;

    json["size_bytes"] = size_bytes;

    return json;
  }
};

struct PrefixStatistics : IndexStatistics {
  CountItem<> length_accesses;
  CountItem<> token_accesses;
  CountItem<> position_accesses;
  int64_t group_count{};
  HistItem<2> group_histogram;

  PrefixStatistics& operator+=(const IndexStatistics& rhs) override {
    IndexStatistics::operator+=(rhs);

    if (auto prhs = dynamic_cast<const PrefixStatistics*>(&rhs)) {
      length_accesses.value += prhs->length_accesses.value;
      token_accesses.value += prhs->token_accesses.value;
      position_accesses.value += prhs->position_accesses.value;
    }

    return *this;
  }

  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json = IndexStatistics::to_json();

    length_accesses.add_to_json("length_accesses", json);
    token_accesses.add_to_json("token_accesses", json);
    position_accesses.add_to_json("position_accesses", json);
    json["group_count"] = group_count;
    group_histogram.add_to_json("group_histogram", json);

    return json;
  }
};

struct PartitionStatistics : IndexStatistics {
  [[nodiscard]] nlohmann::json to_json() const override {
    nlohmann::json json = IndexStatistics::to_json();

    return json;
  }
};

}  // namespace statistics

#endif  // KNN_PROJECT_INDEX_STATISTICS_HH
