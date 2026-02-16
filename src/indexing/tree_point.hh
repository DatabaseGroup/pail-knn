#ifndef KNN_PROJECT_TREE_POINT_HH
#define KNN_PROJECT_TREE_POINT_HH

#include <boost/geometry.hpp>

#include "../similarity/Similarity.hh"
#include "../statistics/transformation_statistics.hh"
#include "../types/types.hh"
#include "../util/set_ops.hh"

namespace boost::geometry::traits {

template <>
struct tag<indexing::tree_point> {
  using type = point_tag;
};

template <>
struct dimension<indexing::tree_point> : boost::mpl::int_<indexing::tree_point::DIMENSIONALITY> {};

template <>
struct coordinate_type<indexing::tree_point> {
  using type = uint16_t;
};

template <>
struct coordinate_system<indexing::tree_point> {
  using type = boost::geometry::cs::cartesian;
};

template <std::size_t Index>
struct access<indexing::tree_point, Index> {
  static_assert(Index < indexing::tree_point::DIMENSIONALITY, "Out of range");
  using Point = indexing::tree_point;
  using CoordinateType = typename coordinate_type<Point>::type;
  static inline CoordinateType get(Point const& p) {
    if constexpr (Index < indexing::tree_point::DIMENSIONALITY / 2) {
      return p.coordinates.f1.arr[Index];
    } else {
      return p.coordinates.f2.arr[Index - indexing::tree_point::DIMENSIONALITY / 2];
    }
  }
  static inline void set(Point& p, CoordinateType const& value) {
    if constexpr (Index < indexing::tree_point::DIMENSIONALITY / 2) {
      p.coordinates.f1.arr[Index] = value;
    } else {
      p.coordinates.f2.arr[Index - indexing::tree_point::DIMENSIONALITY / 2] = value;
    }
  }
};

}  // namespace boost::geometry::traits

namespace boost::geometry::index::detail {

template <typename Strategy>
struct calculate_distance<predicates::nearest<indexing::tree_point>, indexing::tree_box, Strategy, bounds_tag> {
  typedef double result_type;

  static inline bool apply(predicates::nearest<indexing::tree_point> const& p,
                           indexing::tree_box const& b,
                           [[maybe_unused]] Strategy const& s,
                           double& result) {
    result = similarity::JaccardSimilarity::box_distance(p.point_or_relation, b);
    return true;
  }
};

template <typename Strategy>
struct calculate_distance<predicates::nearest<indexing::tree_point>, indexing::tree_point, Strategy, value_tag> {
  typedef double result_type;

  static inline bool apply(predicates::nearest<indexing::tree_point> const& p1,
                           indexing::tree_point const& p2,
                           [[maybe_unused]] Strategy const& strategy,
                           double& result,
                           double current_distance_threshold) {
    auto threshold = 1 - current_distance_threshold;
    auto& r = p1.point_or_relation.record;
    auto& s = p2.record;

    auto r_size = static_cast<int32_t>(r.tokens.size());
    auto s_size = static_cast<int32_t>(s.tokens.size());

    auto& statistics = knn::transformation::TRANSFORMATION_STATISTICS.local().statistics;

    statistics.precandidates.inc();

    auto overlap_threshold = similarity::JaccardSimilarity::eqo(r_size, s_size, threshold);

    auto max_overlap = similarity::max_overlap(p1.point_or_relation.coordinates, p2.coordinates);
    if (max_overlap < overlap_threshold) {
      return false;
    }

    auto overlap = ::util::falsify_or_compute(r.tokens, s.tokens, overlap_threshold);
    statistics.verifications.inc();

    if (overlap > 0) {
      statistics.successful_verifications.inc();
      result = 1 - similarity::JaccardSimilarity::sim(static_cast<int32_t>(p1.point_or_relation.record.tokens.size()),
                                                      static_cast<int32_t>(p2.record.tokens.size()),
                                                      overlap);
      return true;
    }
    return false;
  }
};

}  // namespace boost::geometry::index::detail

#endif  // KNN_PROJECT_TREE_POINT_HH
