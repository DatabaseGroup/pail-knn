#ifndef KNN_PROJECT_TRANSFORMATION_HH
#define KNN_PROJECT_TRANSFORMATION_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <boost/range/irange.hpp>

#include "../../data/set_data.hh"
#include "../../indexing/tree_point.hh"
#include "../../statistics/transformation_statistics.hh"
#include "../../timing/join_timing.hh"
#include "../../types/types.hh"
#include "../multithreaded_join.hh"

namespace bgi = boost::geometry::index;
namespace knn::transformation {

class access_transformation {
public:
  access_transformation() = default;
  explicit access_transformation(const size_t universe_size) : trans(universe_size + 1, 0) {}

  uint8_t operator()(size_t i) { return trans[i]; }

  std::vector<uint8_t> trans;
};

class build_transformation {
public:
  explicit build_transformation(uint8_t m) : groups(m) {}

  std::vector<Tokens> groups;

  access_transformation to_access_transformation(size_t universe_size) const {
    access_transformation at{universe_size};

    for (uint8_t group = 0; group < static_cast<uint8_t>(groups.size()); ++group) {
      for (auto token : groups[group]) {
        at.trans[token] = group;
      }
    }

    return at;
  }
};

struct group_prio {
  uint8_t group_id;
  size_t total_frequency = 0;

  explicit group_prio(uint8_t groupId) : group_id(groupId) {}
  bool operator<(const group_prio& o) const { return this->total_frequency < o.total_frequency; }
  bool operator>(const group_prio& o) const { return this->total_frequency > o.total_frequency; }
};

template <class OrderedTokens, class TokenFrequencies>
void greedy_grouping(build_transformation& t, uint8_t m, OrderedTokens& tokens, TokenFrequencies& token_frequencies) {
  std::vector<group_prio> heap;

  for (uint8_t i = 0; i < m; ++i) {
    heap.emplace_back(i);
  }

  for (auto token : tokens) {
    std::pop_heap(heap.begin(), heap.end(), std::greater{});
    auto group_id = heap.back().group_id;

    t.groups[group_id].push_back(token);
    heap.back().total_frequency += token_frequencies[token];

    std::push_heap(heap.begin(), heap.end(), std::greater{});
  }
}

inline void dual_greedy_grouping(build_transformation& g,
                                 build_transformation& h,
                                 uint8_t m,
                                 std::vector<int32_t>& token_frequencies) {
  auto all_ordered_tokens = boost::irange(static_cast<int32_t>(token_frequencies.size()) - 1, 0, -1);
  greedy_grouping(g, m, all_ordered_tokens, token_frequencies);

  std::vector<build_transformation> tau;

  for (Tokens& group : g.groups) {
    auto& k = tau.emplace_back(m);
    greedy_grouping(k, m, group, token_frequencies);
  }

  std::vector<group_prio> group_prio;
  for (uint8_t i = 0; i < m; ++i) {
    group_prio.emplace_back(i);
  }

  for (auto& trans : tau) {
    std::sort(group_prio.begin(), group_prio.end());

    for (uint8_t i = 0; i < m; ++i) {
      auto& t_group = trans.groups[i];
      auto& h_group = h.groups[group_prio[i].group_id];

      for (auto token : t_group) {
        h_group.push_back(token);
        group_prio[i].total_frequency += token_frequencies[token];
      }
    }
  }
}

class DualTransformer {
public:
  static constexpr uint8_t DIMENSIONALITY = 16;  // == 256 / 16

  explicit DualTransformer(std::vector<int32_t>& token_frequencies) {
    auto universe_size = token_frequencies.size();

    build_transformation b1(DIMENSIONALITY), b2(DIMENSIONALITY);
    dual_greedy_grouping(b1, b2, DIMENSIONALITY, token_frequencies);
    t1 = b1.to_access_transformation(universe_size);
    t2 = b2.to_access_transformation(universe_size);
  }

  void transform(const Records& records, indexing::dfingerprints& fingerprints) {
    for (size_t i = 0; i < records.size(); ++i) {
      transform(records[i], fingerprints[i]);
    }
  }

  void transform(const Record& r, indexing::dual_fingerprint& df) {
    for (const auto token : r.tokens) {
      ++df.f1.arr[t1(token)];
      ++df.f2.arr[t2(token)];
    }
  }

private:
  access_transformation t1;
  access_transformation t2;
};

namespace bgi = boost::geometry::index;

struct DummyState {};

class Transformation : public join::MultithreadedJoin<Transformation, DummyState> {
public:
  using Base = join::MultithreadedJoin<Transformation, BaselineThreadState>;

public:
  using RTree = bgi::rtree<indexing::tree_point, bgi::linear<64>>;

public:
  Transformation(data::SetData& data, int32_t k) : MultithreadedJoin(k, data) {
    auto frequencies = data.count_frequency();
    DualTransformer transformer{frequencies};

    points.reserve(records.size());
    for (int32_t record_id : boost::irange(static_cast<int32_t>(records.size()))) {
      auto& record = records[record_id];
      indexing::dual_fingerprint fp;
      transformer.transform(record, fp);
      points.emplace_back(record_id, record, fp);
    }

    tree = RTree(points);
  }

  nlohmann::json get_json_statistics() const {
    nlohmann::json result;
    statistics::TransformationStatistics final_result;
    for (auto& s : TRANSFORMATION_STATISTICS) {
      final_result += s.statistics;
    }

    result["join"] = final_result.to_json();
    result["timing"] = timing.to_json();

    return result;
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    oneapi::tbb::task_arena arena(concurrency);
    oneapi::tbb::blocked_range<Iterator> iter(begin, end);

    timing.join_time.start();
    arena.execute([&] {
      oneapi::tbb::parallel_for(iter, [&](const tbb::blocked_range<Iterator>& r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
          auto record_id = *it;
          lookup(record_id);
        }
      });
    });
    timing.join_time.stop();
  }

private:
  void lookup(int32_t record_id) {
    auto& point = points[record_id];
    std::vector<indexing::tree_point> result;

    tree.query(bgi::nearest(point, k), std::back_inserter(result));

    /* for (auto& p : result) {
      std::cout << p.record_id << " ";
    }
    std::cout << std::endl;*/

    // compare(record_id, result);
  }

  void compare(int32_t record_id, std::vector<indexing::tree_point>& result) {
    auto& record = records[record_id];
    TopKHeap ref(k);
    auto r_size = static_cast<int32_t>(record.tokens.size());

    for (int32_t cand_id = 0; cand_id < static_cast<int32_t>(records.size()); ++cand_id) {
      auto& s = records[cand_id];
      auto overlap_threshold =
        similarity::JaccardSimilarity::eqo(r_size, static_cast<int32_t>(s.tokens.size()), ref.get_threshold());

      int32_t ovlp = util::falsify_or_compute(record.tokens, s.tokens, overlap_threshold);

      if (ovlp > 0) {
        auto sim = similarity::JaccardSimilarity::sim(r_size, static_cast<int32_t>(s.tokens.size()), ovlp);

        ref.push(cand_id, sim);
      }
    }

    auto& result_first = result.front().record;
    [[maybe_unused]] auto result_lowest =
      similarity::JaccardSimilarity::sim(r_size,
                                         static_cast<int32_t>(result_first.tokens.size()),
                                         util::exact_overlap(record.tokens, result_first.tokens));
    if (static_cast<int32_t>(result.size()) != k) {
      result_lowest = 0;
    }
    assert(result_lowest == ref.get_threshold());
  }

private:
  std::vector<indexing::tree_point> points;
  RTree tree;
};

}  // namespace knn::transformation

namespace boost::geometry::index::detail::rtree::visitors {

template <typename MembersHolder>
class distance_query<MembersHolder, predicates::nearest<indexing::tree_point>> {
  typedef typename MembersHolder::value_type value_type;
  typedef typename MembersHolder::box_type box_type;
  typedef typename MembersHolder::parameters_type parameters_type;
  typedef typename MembersHolder::translator_type translator_type;

  typedef typename index::detail::strategy_type<parameters_type>::type strategy_type;

  typedef typename MembersHolder::node node;
  typedef typename MembersHolder::internal_node internal_node;
  typedef typename MembersHolder::leaf leaf;

  typedef index::detail::predicates_element<
    index::detail::predicates_find_distance<predicates::nearest<indexing::tree_point>>::value,
    predicates::nearest<indexing::tree_point>>
    nearest_predicate_access;
  typedef typename nearest_predicate_access::type nearest_predicate_type;
  typedef typename indexable_type<translator_type>::type indexable_type;

  typedef index::detail::calculate_distance<nearest_predicate_type, indexable_type, strategy_type, value_tag>
    calculate_value_distance;
  typedef index::detail::calculate_distance<nearest_predicate_type, box_type, strategy_type, bounds_tag>
    calculate_node_distance;
  typedef typename calculate_value_distance::result_type value_distance_type;
  typedef typename calculate_node_distance::result_type node_distance_type;

  typedef typename MembersHolder::size_type size_type;
  typedef typename MembersHolder::node_pointer node_pointer;

  using neighbor_data = std::pair<value_distance_type, const value_type*>;
  using neighbors_type = std::vector<neighbor_data>;

  struct branch_data {
    branch_data(node_distance_type d, size_type rl, node_pointer p) : distance(d), reverse_level(rl), ptr(p) {}

    node_distance_type distance;
    size_type reverse_level;
    node_pointer ptr;
  };
  using branches_type = priority_queue<branch_data, branch_data_comp>;

public:
  distance_query(MembersHolder const& members, predicates::nearest<indexing::tree_point> const& pred)
      : m_tr(members.translator()), m_strategy(index::detail::get_strategy(members.parameters())), m_pred(pred) {
    m_neighbors.reserve((std::min)(members.values_count, size_type(max_count())));
    // m_branches.reserve(members.parameters().get_min_elements() * members.leafs_level); ?
    //  min, max or average?
  }

  template <typename OutIter>
  size_type apply(MembersHolder const& members, OutIter out_it) {
    return apply(members.root, members.leafs_level, out_it);
  }

private:
  template <typename OutIter>
  size_type apply(node_pointer ptr, size_type reverse_level, OutIter out_it) {
    namespace id = index::detail;

    if (max_count() <= 0) {
      return 0;
    }

    int64_t verifications = 0;
    int64_t heap_pops = 1;

    auto& statistics = knn::transformation::TRANSFORMATION_STATISTICS.local().statistics;

    for (;;) {
      if (reverse_level > 0) {
        internal_node& n = rtree::get<internal_node>(*ptr);
        // fill array of nodes meeting predicates
        for (auto const& p : rtree::elements(n)) {
          node_distance_type node_distance;  // for distance predicate

          statistics.tree_accesses.inc();

          // if current node meets predicates (0 is dummy value)
          if (id::predicates_check<id::bounds_tag>(m_pred, 0, p.first, m_strategy)
              // and if distance is ok
              && calculate_node_distance::apply(predicate(), p.first, m_strategy, node_distance)
              // and if current node is closer than the furthest neighbor
              && !ignore_branch(node_distance)) {
            // add current node's data into the list
            m_branches.push(branch_data(node_distance, reverse_level - 1, p.second));
          }
        }
      } else {
        leaf& n = rtree::get<leaf>(*ptr);
        // search leaf for closest value meeting predicates
        for (auto const& v : rtree::elements(n)) {
          value_distance_type value_distance;  // for distance predicate

          ++verifications;
          // if value meets predicates
          if (id::predicates_check<id::value_tag>(m_pred, v, m_tr(v), m_strategy)
              // and if distance is ok
              &&
              calculate_value_distance::apply(predicate(), m_tr(v), m_strategy, value_distance, current_threshold())) {
            // store value
            store_value(value_distance, boost::addressof(v));
          }
        }
      }

      if (m_branches.empty() || ignore_branch(m_branches.top().distance)) {
        break;
      }

      ptr = m_branches.top().ptr;
      reverse_level = m_branches.top().reverse_level;
      m_branches.pop();
      ++heap_pops;
    }

    statistics.heap_pops.record(heap_pops);
    statistics.avg_verifications.record(verifications);
    statistics.final_k_similarity.record(1 - current_threshold());
    statistics.final_heap_size.record(static_cast<int64_t>(m_branches.size()));

    for (auto const& p : m_neighbors) {
      *out_it = *(p.second);
      ++out_it;
    }

    return m_neighbors.size();
  }

  bool ignore_branch(node_distance_type const& node_distance) const {
    return m_neighbors.size() == max_count() && m_neighbors.front().first <= node_distance;
  }

  node_distance_type current_threshold() const {
    return m_neighbors.size() == max_count() ? m_neighbors.front().first : node_distance_type(1);
  }

  void store_value(value_distance_type value_distance, const value_type* value_ptr) {
    if (m_neighbors.size() < max_count()) {
      m_neighbors.push_back(std::make_pair(value_distance, value_ptr));

      if (m_neighbors.size() == max_count()) {
        std::make_heap(m_neighbors.begin(), m_neighbors.end(), pair_first_less());
      }
    } else if (value_distance < m_neighbors.front().first) {
      std::pop_heap(m_neighbors.begin(), m_neighbors.end(), pair_first_less());
      m_neighbors.back() = std::make_pair(value_distance, value_ptr);
      std::push_heap(m_neighbors.begin(), m_neighbors.end(), pair_first_less());
    }
  }

  std::size_t max_count() const { return nearest_predicate_access::get(m_pred).count; }

  nearest_predicate_type const& predicate() const { return nearest_predicate_access::get(m_pred); }

  translator_type const& m_tr;
  strategy_type m_strategy;

  predicates::nearest<indexing::tree_point> const& m_pred;

  branches_type m_branches;
  neighbors_type m_neighbors;
};

}  // namespace boost::geometry::index::detail::rtree::visitors

#endif  // KNN_PROJECT_TRANSFORMATION_HH
