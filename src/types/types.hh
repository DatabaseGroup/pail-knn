#ifndef KNN_PROJECT_TYPES_HH
#define KNN_PROJECT_TYPES_HH

#include <boost/align/aligned_allocator.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/strategies/relate/cartesian.hpp>
#include <vector>

#include "absl/container/flat_hash_map.h"

using RecordSize = int32_t;
using RecordId = int32_t;
using RecordIds = std::vector<RecordId>;
using Candidates = RecordIds;
using CandidateBitmap = std::vector<bool>;
using Token = int32_t;
using TokenPosition = int32_t;
using Tokens = std::vector<Token>;

struct Record {
  Tokens tokens;

  explicit Record(Tokens&& tokens) : tokens(std::forward<Tokens>(tokens)) {}
  friend std::ostream& operator<<(std::ostream& os, const Record& record) {
    os << "[";
    for (size_t i = 0; i < record.tokens.size(); ++i) {
      os << record.tokens[i];
      if (i < record.tokens.size() - 1) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }
};

using Records = std::vector<Record>;

template <class K, class V>
using HashTable = absl::flat_hash_map<K, V>;

struct cand_entry {
  RecordId overlap = 0;
  int32_t p_r = 0;
  int32_t p_s = 0;
};

using CandidateOverlaps = std::vector<cand_entry>;

namespace indexing {

struct fingerprint {
  static constexpr int32_t DIMENSIONALITY = 16;

  fingerprint() { std::memset(&arr, 0, sizeof(arr)); }

  uint16_t arr[DIMENSIONALITY];
};

struct dual_fingerprint {
  static constexpr int32_t DIMENSIONALITY = 2 * fingerprint::DIMENSIONALITY;

  fingerprint f1;
  fingerprint f2;

  friend std::ostream& operator<<(std::ostream& os, const dual_fingerprint& dt) {
    os << "[ [";
    for (uint16_t a : dt.f1.arr) {
      os << a << ", ";
    }
    os << "]\n  [";
    for (uint16_t a : dt.f2.arr) {
      os << a << ", ";
    }
    os << "]\n";
    return os;
  }
};

using dfingerprints =
  std::vector<dual_fingerprint, boost::alignment::aligned_allocator<dual_fingerprint, sizeof(dual_fingerprint)>>;

struct tree_point {
  static constexpr int32_t DIMENSIONALITY = dual_fingerprint::DIMENSIONALITY;

  tree_point(const RecordId recordId, const Record& record, const dual_fingerprint& coordinates)
      : record_id(recordId), record(record), coordinates(coordinates) {}

  const RecordId record_id;
  const Record& record;
  dual_fingerprint coordinates;
};

using tree_box = boost::geometry::model::box<
  boost::geometry::model::point<::uint16_t, tree_point::DIMENSIONALITY, boost::geometry::cs::cartesian>>;

}  // namespace indexing

#endif  // KNN_PROJECT_TYPES_HH
