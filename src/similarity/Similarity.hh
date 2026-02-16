#ifndef KNN_PROJECT_SIMILARITY_HH
#define KNN_PROJECT_SIMILARITY_HH

#include <boost/geometry.hpp>

#include "../indexing/length_group.hh"
#include "../types/types.hh"

namespace bg = boost::geometry;

namespace similarity {

#ifdef __x86_64__
inline int32_t max_overlap(const indexing::dual_fingerprint& fp1, const indexing::dual_fingerprint& fp2) {
  __m256i min1 = _mm256_min_epu16(_mm256_loadu_si256(reinterpret_cast<__m256i const*>(fp1.f1.arr)),
                                  _mm256_loadu_si256(reinterpret_cast<__m256i const*>(fp2.f1.arr)));
  __m256i min2 = _mm256_min_epu16(_mm256_loadu_si256(reinterpret_cast<__m256i const*>(fp1.f2.arr)),
                                  _mm256_loadu_si256(reinterpret_cast<__m256i const*>(fp2.f2.arr)));

  // horizontally add two neighboring 16-bit integers (a vector now contains 8 * 32-bit integers)
  __m256i sum32_1 = _mm256_madd_epi16(min1, _mm256_set1_epi16(1));
  __m256i sum32_2 = _mm256_madd_epi16(min2, _mm256_set1_epi16(1));

  // add upper 4 * 32-bit integers to lower 4 * 32-bit integers
  __m128i sum64_1 = _mm_add_epi32(_mm256_castsi256_si128(sum32_1), _mm256_extracti128_si256(sum32_1, 1));
  __m128i sum64_2 = _mm_add_epi32(_mm256_castsi256_si128(sum32_2), _mm256_extracti128_si256(sum32_2, 1));

  // horizontal sum of 4 32-bit integers stored in 128-bit register is not that nice
  // idea: move upper 64-bit to lower 64-bit and add (2 * 32-bit numbers remain)
  //       swap the two remaining numbers and add again (1 * 32-bit number remains)
  //       copy the lower 32-bits out of the register as the result
  __m128i upper64_1 = _mm_unpackhi_epi64(sum64_1, sum64_1);
  __m128i upper64_2 = _mm_unpackhi_epi64(sum64_2, sum64_2);
  __m128i sum128_1 = _mm_add_epi32(sum64_1, upper64_1);
  __m128i sum128_2 = _mm_add_epi32(sum64_2, upper64_2);
  __m128i upper32_1 = _mm_shuffle_epi32(sum128_1, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i upper32_2 = _mm_shuffle_epi32(sum128_2, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sum256_1 = _mm_add_epi32(sum128_1, upper32_1);
  __m128i sum256_2 = _mm_add_epi32(sum128_2, upper32_2);

  auto res1 = _mm_cvtsi128_si32(sum256_1);
  auto res2 = _mm_cvtsi128_si32(sum256_2);

  return std::min(res1, res2);
}
#else
inline int32_t fp_overlap(const indexing::fingerprint& fp1, const indexing::fingerprint& fp2) {
  int32_t sum = 0;
  for (size_t i = 0; i < indexing::fingerprint::DIMENSIONALITY; ++i) {
    sum += std::min(fp1.arr[i], fp2.arr[i]);
  }
  return sum;
}

inline int32_t max_overlap(const indexing::dual_fingerprint& fp1, const indexing::dual_fingerprint& fp2) {
  return std::min(fp_overlap(fp1.f1, fp2.f1), fp_overlap(fp1.f2, fp2.f2));
}
#endif

class JaccardSimilarity {
public:
  static double sim(int32_t r_size, int32_t s_size, int32_t overlap) {
    return static_cast<double>(overlap) / (r_size + s_size - overlap);
  }

  static double sim_ub(int32_t r_size, int32_t s_size, int32_t p_r, int32_t p_s) {
    auto min = std::min(r_size - p_r, s_size - p_s);

    return min / static_cast<double>((r_size + s_size - min));
  }

  template <class LGroup>
  static double grouped_sim_ub(LGroup& grouping, int32_t r_size, indexing::lengrp_idx lengrp_ptr, int32_t p_r, [[maybe_unused]] int32_t p_s) {
    auto s_lb = grouping.lower_bound(lengrp_ptr);

    int32_t min_size = std::max(s_lb,r_size - p_r);
    return sim_ub(r_size, min_size, p_r, min_size-r_size+p_r);
  }

  static int32_t eqo(int32_t r_size, int32_t s_size, double threshold) {
    return static_cast<int32_t>(std::ceil((r_size + s_size) * threshold / (1 + threshold)));
  }

  static int32_t max_ps(int32_t r_size, int32_t s_size, double threshold) {
    return static_cast<int32_t>(std::floor(s_size - (threshold / (1 + threshold)) * (r_size + s_size)));
  }

  template <class LGroup>
  static int32_t max_grouped_ps(LGroup& grouping, int32_t r_size, indexing::lengrp_idx lengrp_ptr, double threshold) {
    return max_ps(r_size, grouping.upper_bound(lengrp_ptr), threshold);
  }

  static std::pair<int32_t, int32_t> s_size_bounds(int32_t r_size, int32_t p_r, int32_t p_s, double threshold) {
    return {std::ceil((r_size - (1 + threshold) * p_r) / threshold),
            std::floor(r_size * threshold + (1 + threshold) * p_s)};
  }

  static double point_distance(const indexing::tree_point& p1, const indexing::tree_point& p2) {
    auto m_overlap = static_cast<double>(max_overlap(p1.coordinates, p2.coordinates));
    auto p1_size = static_cast<double>(p1.record.tokens.size());
    auto p2_size = static_cast<double>(p2.record.tokens.size());

    return 1. - (m_overlap / (p1_size + p2_size - m_overlap));
  }

#ifdef __x86_64__
  static double box_distance(const indexing::tree_point& point, const indexing::tree_box& box) {
    double bound;
    {
      __m256i p = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(point.coordinates.f1.arr));
      __m256i b_bot = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&box.min_corner().get<0>()));
      __m256i b_top = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&box.max_corner().get<0>()));

      auto num = static_cast<double>(box_numerator(p, b_top));
      auto den = static_cast<double>(box_denominator(p, b_bot));

      bound = 1. - 1. * num / den;
    }
    {
      __m256i p = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(point.coordinates.f2.arr));
      __m256i b_bot = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&box.min_corner().get<16>()));
      __m256i b_top = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&box.max_corner().get<16>()));

      auto num = static_cast<double>(box_numerator(p, b_top));
      auto den = static_cast<double>(box_denominator(p, b_bot));

      bound = std::max(bound, 1. - 1. * num / den);
    }

    return bound;
  }

  static int32_t box_denominator(__m256i point, __m256i box_bot) {
    auto max = _mm256_max_epu16(point, box_bot);

    return util::hsum_256_epu16(max);
  }

  static int32_t box_numerator(__m256i point, __m256i box_top) {
    auto min = _mm256_min_epu16(point, box_top);

    return util::hsum_256_epu16(min);
  }
#else
  static double box_distance(const indexing::tree_point& point, const indexing::tree_box& box) {
    auto box1_top = &box.max_corner().get<0>();
    auto box1_bot = &box.min_corner().get<0>();
    auto bound1 = 1. - 1. * box_numerator(point.coordinates.f1, box1_top) / box_denominator(point.coordinates.f1, box1_bot);

    auto box2_top = &box.max_corner().get<indexing::fingerprint::DIMENSIONALITY>();
    auto box2_bot = &box.min_corner().get<indexing::fingerprint::DIMENSIONALITY>();
    auto bound2 = 1. - 1. * box_numerator(point.coordinates.f2, box2_top) / box_denominator(point.coordinates.f2, box2_bot);

    return std::min(bound1, bound2);
  }

  static int32_t box_denominator(const indexing::fingerprint& point, const uint16_t* const box_bot) {
    int32_t sum = 0;
    for (size_t i = 0; i < indexing::fingerprint::DIMENSIONALITY; ++i) {
      sum += std::max(point.arr[i], box_bot[i]);
    }
    return sum;
  }

  static int32_t box_numerator(const indexing::fingerprint& point, const uint16_t* const box_top) {
    int32_t sum = 0;
    for (size_t i = 0; i < indexing::fingerprint::DIMENSIONALITY; ++i) {
      sum += std::min(point.arr[i], box_top[i]);
    }
    return sum;
  }
#endif
};

}  // namespace similarity

#endif  // KNN_PROJECT_SIMILARITY_HH
