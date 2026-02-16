#ifndef KNN_PROJECT_EQUEUE_HH
#define KNN_PROJECT_EQUEUE_HH

#include <queue>

namespace knn {

struct Event {
  int32_t p_r;
  indexing::lengrp_idx lengrp_ptr;
  int32_t p_s;
  double sim_ub;

  bool operator<(const Event& rhs) const { return sim_ub < rhs.sim_ub; }

  bool operator>(const Event& rhs) const { return sim_ub > rhs.sim_ub; }

  Event(int32_t p_r, int32_t s_grp, int32_t p_s, double sim_ub)
      : p_r(p_r), lengrp_ptr(s_grp), p_s(p_s), sim_ub(sim_ub) {}

  template <typename H>
  friend H AbslHashValue(H h, const Event& c) {
    return H::combine(std::move(h), c.p_r, c.lengrp_ptr, c.p_s);
  }

  bool operator==(const Event& rhs) const { return p_r == rhs.p_r && lengrp_ptr == rhs.lengrp_ptr && p_s == rhs.p_s; }
  bool operator!=(const Event& rhs) const { return !(rhs == *this); }
};

template <class LGrouping>
class EQueue {
public:
  explicit EQueue(const LGrouping& l_grouping) : l_grouping(l_grouping) {}

public:
  int32_t initialize(int32_t r_size) {
    auto r_grp_it = std::lower_bound(l_grouping.get_groups().begin(),
                                     l_grouping.get_groups().end(),
                                     indexing::length_group{r_size},
                                     [&](const indexing::length_group& g1, const indexing::length_group& g2) {
                                       return g1.lower_bound <= g2.lower_bound;
                                     });
    auto r_grp = std::distance(l_grouping.get_groups().begin(), r_grp_it) - 1;
    enqueue(r_size, 0, r_grp, l_grouping.upper_bound(r_grp) - r_size);
    // - 1 for being at most the last group, - 1 as the last group is a tombstone
    if (r_grp < static_cast<int32_t>(l_grouping.size()) - 2) {
      enqueue(r_size, 0, r_grp + 1, l_grouping.upper_bound(r_grp + 1) - r_size);
    }
    if (r_grp > 0) {
      enqueue(r_size, r_size - l_grouping.upper_bound(r_grp - 1), r_grp - 1, 0);
    }

    return r_grp;
  }

  void enqueue_next(int32_t r_size, indexing::lengrp_idx s_grp, Event& e) {
    if (l_grouping.lower_bound(s_grp) > r_size) {
      enqueue(r_size, e.p_r + 1, e.lengrp_ptr, e.p_s + 1);
      // - 1 for being at most the last group, - 1 as the last group is a tombstone
      if (e.p_r == 0 && e.lengrp_ptr < static_cast<int32_t>(l_grouping.size()) - 2) {
        enqueue(r_size, e.p_r, e.lengrp_ptr + 1, inc_pos(e.p_s, e.lengrp_ptr));
      }
    } else if (l_grouping.upper_bound(s_grp) < r_size) {
      enqueue(r_size, e.p_r + 1, e.lengrp_ptr, e.p_s + 1);
      if (e.p_s == 0 && e.lengrp_ptr > 0) {
        enqueue(r_size, dec_pos(e.p_r, e.lengrp_ptr), e.lengrp_ptr - 1, e.p_s);
      }
    } else {
      enqueue(r_size, e.p_r + 1, e.lengrp_ptr, e.p_s + 1);
    }
  }

  bool probe_vertical([[maybe_unused]] int32_t r_size, int32_t r_size_ptr, const Event& e) {
    if (r_size_ptr > e.lengrp_ptr) {
      return e.p_s == 0;
    }
    return false;
  }

  Event pop() {
    Event e = queue.top();
    queue.pop();

    return e;
  }

  void clear() { queue = std::priority_queue<Event>(); }

  [[nodiscard]] size_t size() const { return queue.size(); }

  [[nodiscard]] bool empty() const { return size() == 0; }

private:
  void enqueue(int32_t r_size, int32_t p_r, int32_t s_size_ptr, int32_t p_s) {
    const Event e{
      p_r, s_size_ptr, p_s, similarity::JaccardSimilarity::grouped_sim_ub(l_grouping, r_size, s_size_ptr, p_r, p_s)};
    queue.push(e);
  }

  int32_t inc_pos(int32_t p_r, indexing::lengrp_idx s_grp) {
    int32_t size_diff = l_grouping.lower_bound(s_grp + 1) - l_grouping.lower_bound(s_grp);
    return p_r + size_diff;
  }

  int32_t dec_pos(int32_t p_r, indexing::lengrp_idx s_grp) {
    int32_t size_diff = l_grouping.upper_bound(s_grp) - l_grouping.upper_bound(s_grp - 1);
    return p_r + size_diff;
  }

private:
  std::priority_queue<Event> queue;
  const LGrouping l_grouping;
};

}  // namespace knn

#endif  // KNN_PROJECT_EQUEUE_HH
