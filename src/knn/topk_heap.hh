#ifndef KNN_PROJECT_TOPK_HEAP_HH
#define KNN_PROJECT_TOPK_HEAP_HH

class TopKHeap {
public:
  explicit TopKHeap(int32_t k) {
    heap.resize(k);
    this->minimum = 0;
  }

  void push(int32_t set_id, double similarity) {
    std::pop_heap(heap.begin(), heap.end(), std::greater{});
    heap.back().set_id = set_id;
    heap.back().similarity = similarity;
    std::push_heap(heap.begin(), heap.end(), std::greater{});

    this->minimum = heap.front().similarity;
  }

  [[nodiscard]] double get_threshold() const { return this->minimum; }

  [[nodiscard]] RecordIds get_result() const {
    RecordIds res;
    res.reserve(heap.size());
    std::ranges::for_each(heap, [&](auto element) { res.push_back(element.set_id); });
    return res;
  }

private:
  struct heap_element {
    int32_t set_id = -1;
    double similarity = 0;

    bool operator<(const heap_element& rhs) const { return similarity < rhs.similarity; }
    bool operator>(const heap_element& rhs) const { return similarity > rhs.similarity; }
  };

private:
  std::vector<heap_element> heap;
  double minimum;
};

#endif  // KNN_PROJECT_TOPK_HEAP_HH
