#ifndef KNN_PROJECT_SET_OPS_HH
#define KNN_PROJECT_SET_OPS_HH

#if __x86_64__
#include <immintrin.h>
#endif

#include "../types/types.hh"

namespace util {

bool inline verify_pair(const Tokens& r1,
                        const Tokens& r2,
                        int32_t overlap_threshold,
                        int32_t position_r1,
                        int32_t position_r2,
                        int32_t found_overlap = 0) {
  auto r1_size = static_cast<int32_t>(r1.size());
  auto r2_size = static_cast<int32_t>(r2.size());
  int32_t max_r1 = r1_size - position_r1 + found_overlap;
  int32_t max_r2 = r2_size - position_r2 + found_overlap;

  while (max_r1 >= overlap_threshold && max_r2 >= overlap_threshold && found_overlap < overlap_threshold) {
    if (r1[position_r1] == r2[position_r2]) {
      ++position_r1;
      ++position_r2;
      ++found_overlap;
    } else if (r1[position_r1] < r2[position_r2]) {
      ++position_r1;
      --max_r1;
    } else {
      ++position_r2;
      --max_r2;
    }
  }
  return found_overlap >= overlap_threshold;
}

inline int32_t exact_overlap(const Tokens& r1, const Tokens& r2) {
  int32_t found_overlap = 0;
  auto r1_size = static_cast<int32_t>(r1.size());
  auto r2_size = static_cast<int32_t>(r2.size());
  int32_t p1 = 0, p2 = 0;

  while (p1 < r1_size && p2 < r2_size) {
    if (r1[p1] == r2[p2]) {
      ++p1;
      ++p2;
      ++found_overlap;
    } else if (r1[p1] < r2[p2]) {
      ++p1;
    } else {
      ++p2;
    }
  }
  return found_overlap;
}

inline int32_t falsify_or_compute(const Tokens& r1,
                                  const Tokens& r2,
                                  int32_t overlap_threshold,
                                  int32_t& p1,
                                  int32_t& p2,
                                  int32_t found_overlap) {
  auto r1_size = static_cast<int32_t>(r1.size());
  auto r2_size = static_cast<int32_t>(r2.size());
  int32_t max_r1 = r1_size - p1 + found_overlap;
  int32_t max_r2 = r2_size - p2 + found_overlap;

  while (p1 < r1_size && p2 < r2_size && max_r1 >= overlap_threshold && max_r2 >= overlap_threshold) {
    if (r1[p1] == r2[p2]) {
      ++p1;
      ++p2;
      ++found_overlap;
    } else if (r1[p1] < r2[p2]) {
      ++p1;
      --max_r1;
    } else {
      ++p2;
      --max_r2;
    }
  }
  return found_overlap >= overlap_threshold ? found_overlap : -1;
}

inline int32_t falsify_or_compute(const Tokens& r1, const Tokens& r2, int32_t overlap_threshold) {
  int32_t p1 = 0;
  int32_t p2 = 0;
  return falsify_or_compute(r1, r2, overlap_threshold, p1, p2, 0);
}

static uint64_t log2(const uint64_t x) { return (63 - __builtin_clzll(x)); }

// branchless binary search
template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
size_t binary_search(const T* arr, size_t size, const T key) {
  intptr_t pos = -1;
  // we need the number of iterations to guarantee finding the value if it exists, which is
  // log2(size + 1) = k; this is the position of the first 1 bit in size (counting from LSB)
  auto logstep = static_cast<intptr_t>(log2(size));
  // first step makes the array of size 2^k for some k
  intptr_t step = intptr_t(1) << logstep;
  pos = (arr[pos + step] < key ? size - step : pos);
  step >>= 1;
  while (step > 0) {
    pos = (arr[pos + step] < key ? pos + step : pos);
    step >>= 1;
  }
  return pos + 1;
}

template <int32_t depth>
int32_t
suffix_filter(const Tokens& r, int32_t r_l, int32_t r_u, const Tokens& s, int32_t s_l, int32_t s_u, int32_t hd_max) {
  auto r_size = r_u - r_l;
  auto s_size = s_u - s_l;
  if constexpr (depth == 0) {
    return std::abs(r_size - s_size);
  } else {
    // std::cout << "searching in r: [";
    // for (int32_t i = r_l; i < r_u; ++i) {
    //   std::cout << r[i] << " ";
    // }
    // std::cout << "], s: [";
    // for (int32_t i = s_l; i < s_u; ++i) {
    //   std::cout << s[i] << " ";
    // }
    // std::cout << "] ";
    if (r_size == 0 || s_size == 0) {
      // std::cout << std::endl;
      return std::abs(r_size - s_size);
    }
    auto s_mid = (s_u + s_l) / 2;
    auto token = s[s_mid];
    // std::cout << "pivot: " << token << std::endl;
    auto r_mid = r_l + static_cast<int32_t>(binary_search(r.data() + r_l, r_u - r_l, token));
    auto diff = token == r[r_mid] ? 0 : 1;

    auto lower_diff = std::abs((r_mid - r_l) - (s_mid - s_l));
    auto upper_diff = std::abs((r_u - r_mid) - (s_u - (s_mid + diff)));

    auto hd_approx = lower_diff + upper_diff + diff;
    if (hd_approx > hd_max) {
      return hd_approx;
    } else {
      auto temp1 = hd_max - lower_diff - diff;
      auto temp2 = hd_max - upper_diff - diff;

      auto hd_l = suffix_filter<depth - 1>(r, r_l, r_mid, s, s_l, s_mid, temp2);
      auto hd_u = suffix_filter<depth - 1>(r, r_mid + (1 - diff), r_u, s, s_mid + 1, s_u, temp1);

      return hd_l + hd_u + diff;
    }
  }
}

#if __x86_64__
int32_t hsum_256_epu16(__m256i a) {
  __m256i sum32_1 = _mm256_madd_epi16(a, _mm256_set1_epi16(1));

  // add upper 4 * 32-bit integers to lower 4 * 32-bit integers
  __m128i sum64_1 = _mm_add_epi32(_mm256_castsi256_si128(sum32_1), _mm256_extracti128_si256(sum32_1, 1));

  // horizontal sum of 4 32-bit integers stored in 128-bit register is not that nice
  // idea: move upper 64-bit to lower 64-bit and add (2 * 32-bit numbers remain)
  //       swap the two remaining numbers and add again (1 * 32-bit number remains)
  //       copy the lower 32-bits out of the register as the result
  __m128i upper64_1 = _mm_unpackhi_epi64(sum64_1, sum64_1);
  __m128i sum128_1 = _mm_add_epi32(sum64_1, upper64_1);
  __m128i upper32_1 = _mm_shuffle_epi32(sum128_1, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sum256_1 = _mm_add_epi32(sum128_1, upper32_1);

  auto res1 = _mm_cvtsi128_si32(sum256_1);

  return res1;
}
#endif

}  // namespace util

#endif  // KNN_PROJECT_SET_OPS_HH
