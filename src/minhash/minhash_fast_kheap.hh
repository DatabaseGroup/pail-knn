#ifndef KNN_PROJECT_MINHASH_FAST_KHEAP_HH
#define KNN_PROJECT_MINHASH_FAST_KHEAP_HH

#include <algorithm>
#include <numeric>
#include <random>

namespace minhash {

class MinHasher {
public:
  explicit MinHasher(int32_t universe_size) {
    order.resize(universe_size + 1);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937{std::random_device{}()});
  }

public:
  int32_t hash(Tokens& tokens) {
    return *std::min_element(tokens.begin(), tokens.end(), [&](Token a, Token b) { return lt(a, b); });
  }

private:
  std::vector<int32_t> order;
  bool lt(Token a, Token b) {
    return order[a] < order[b];
  }
};

}

#endif  // KNN_PROJECT_MINHASH_FAST_KHEAP_HH
