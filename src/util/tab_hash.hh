#ifndef KNN_PROJECT_TAB_HASH_HH
#define KNN_PROJECT_TAB_HASH_HH

#include <random>
namespace util {

class TabHash {
public:
  using HashVal = uint64_t;

public:
  explicit TabHash() : rand(std::random_device{}()){};

  explicit TabHash(size_t universe_size) : rand(std::random_device{}()) { resize(universe_size); }

  void resize(size_t universe_size) {
    random.reserve(universe_size);

    for (size_t i = 0; i < universe_size; ++i) {
      random.push_back(rand());
    }
  }

  void push_back() { random.push_back(rand()); }

  HashVal operator()(uint64_t key) {
    while (key >= random.size()) {
      push_back();
    }

    return random[key];
  }

  HashVal add(HashVal a, int32_t key) { return a ^ operator()(key); }

  HashVal remove(HashVal a, int32_t key) { return a ^ operator()(key); }

private:
  std::vector<HashVal> random;
  std::mt19937 rand;
};

}  // namespace util

#endif  // KNN_PROJECT_TAB_HASH_HH
