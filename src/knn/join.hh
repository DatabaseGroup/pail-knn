#ifndef KNN_PROJECT_JOIN_HH
#define KNN_PROJECT_JOIN_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "../indexing/inv_index.hh"
#include "../similarity/Similarity.hh"
#include "../statistics/join_perf.hh"
#include "../statistics/join_statistics.hh"
#include "../types/types.hh"
#include "../util/set_ops.hh"
#include "baseline.hh"
#include "equeue.hh"
#include "multithreaded_join.hh"
#include "topk_heap.hh"
#include "transformation/transformation.hh"

namespace knn {

enum PositionalMode { TOPK_BASELINE, FILTER, FULL };

struct PosThreadState {
  Candidates candidates;
  CandidateOverlaps overlaps;
  statistics::JoinStatistics statistics;
  statistics::PrefixStatistics index_statistics;
};

template <class Similarity,
          class LGrouping,
          PositionalMode pos_mode = FILTER,
          int32_t suffix_filter_depth = 0,
          int32_t sketch_vec = 0>
class KNNJoin
    : public join::MultithreadedJoin<KNNJoin<Similarity, LGrouping, pos_mode, suffix_filter_depth, sketch_vec>,
                                     PosThreadState> {
public:
  using Base = join::MultithreadedJoin<KNNJoin, PosThreadState>;

public:
  explicit KNNJoin(data::SetData& data, const int32_t k, LGrouping& l_grouping)
      : Base(k, data), index(data, l_grouping), l_grouping(l_grouping) {
    falsification_idx = data.count_frequency();
    index.get_statistics().group_count = static_cast<int64_t>(l_grouping.get_groups().size());
    l_grouping.store_histogram(index.get_statistics().group_histogram);

    if constexpr (sketch_vec > 0) {
      fingerprints.resize(Base::records.size());
      transformation::DualTransformer transformer{falsification_idx};
      transformer.transform(Base::records, fingerprints);
    }
  }

  nlohmann::json get_json_statistics() {
    nlohmann::json json;
    auto index_statistics = Base::get_index_statistics(index.get_statistics(), &PosThreadState::index_statistics);
    json["index"] = index_statistics.to_json();

    nlohmann::json inner = Base::get_join_statistics();
    json["join"] = inner;

    json["timing"] = Base::timing.to_json();

    return json;
  }

  void print_order(int32_t query_size, int32_t until) {
    EQueue<LGrouping> q(l_grouping);

    q.initialize(query_size);
    for (int32_t i = 0; i < until; ++i) {
      auto e = q.pop();

      auto min_size = l_grouping.lower_bound(e.lengrp_ptr);
      auto max_size = l_grouping.upper_bound(e.lengrp_ptr);

      std::cout << (i + 1) << ": (" << e.p_r << ", [" << min_size << ", " << max_size << "], " << e.p_s
                << "): " << e.sim_ub << std::endl;

      q.enqueue_next(query_size, e.lengrp_ptr, e);
    }
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    Base::run_parallel_lookups(
      begin,
      end,
      concurrency,
      [&](PosThreadState& state) { state.overlaps.resize(Base::records.size()); },
      [&](const int32_t record_id, PosThreadState& state) { lookup(record_id, state); },
      [&](const int32_t, PosThreadState& state) {
        for (auto& cand_id : state.candidates) {
          state.overlaps[cand_id].overlap = 0;
        }
        state.candidates.clear();
      });
  }

private:
  void lookup(const int32_t record_id, PosThreadState& state) {
    auto& record = Base::records[record_id];
    int32_t actual_k = sim_above_zero(record);
    auto r_size = static_cast<int32_t>(record.tokens.size());

    auto& candidates = state.candidates;
    auto& overlaps = state.overlaps;

    EQueue<LGrouping> q(l_grouping);
    TopKHeap heap(actual_k);
    auto r_grp = q.initialize(r_size);

    int64_t heap_pops = 0;
    size_t candidates_offset = 0;
    while (!q.empty()) {
      auto e = q.pop();
      ++heap_pops;

      if (e.sim_ub <= heap.get_threshold()) {
        break;
      }

      if constexpr (use_index_pos()) {
        auto p_ub = std::min(e.p_s, Similarity::max_grouped_ps(l_grouping, r_size, e.lengrp_ptr, heap.get_threshold()));
        if (p_ub == e.p_s) {
          index.read_cross(record, e.lengrp_ptr, e.p_r, e.p_s, candidates, overlaps, state.index_statistics);
        } else {
          // todo clean up
          if (q.probe_vertical(r_size, r_grp, e)) {
            index.read_vertical_filtered(
              record, e.lengrp_ptr, e.p_r, p_ub, candidates, overlaps, state.index_statistics);
          } else {
            auto token = record.tokens[e.p_r];
            index.read_horizontal_filtered(
              token, e.lengrp_ptr, e.p_r, p_ub, candidates, overlaps, state.index_statistics);
          }
        }
      } else {
        const bool vertical = q.probe_vertical(r_size, r_grp, e);
        int32_t p_ub;

        if constexpr (pos_mode == FILTER) {
          p_ub = Similarity::max_grouped_ps(l_grouping, r_size, e.lengrp_ptr, heap.get_threshold());
        } else {
          p_ub = std::numeric_limits<int32_t>::max();
        }

        if (vertical) {
          index.read_vertical_filtered(record, e.lengrp_ptr, e.p_r, p_ub, candidates, overlaps, state.index_statistics);
        } else {
          auto token = record.tokens[e.p_r];
          index.read_horizontal_filtered(
            token, e.lengrp_ptr, e.p_r, p_ub, candidates, overlaps, state.index_statistics);
        }
      }

      for (auto it = candidates.begin() + static_cast<int64_t>(candidates_offset); it != candidates.end(); ++it) {
        auto cand_id = *it;
        auto& s = Base::records[cand_id];
        auto s_size = static_cast<int32_t>(s.tokens.size());
        auto overlap_threshold = Similarity::eqo(r_size, static_cast<int32_t>(s_size), heap.get_threshold());
        auto p_r = overlaps[cand_id].p_r;
        auto p_s = overlaps[cand_id].p_s;

        if constexpr (suffix_filter_depth > 0) {
          auto max_hd = r_size + s_size - 2 * (overlap_threshold - overlaps[cand_id].overlap) - (p_r + p_s + 2);
          auto approx_hd =
            util::suffix_filter<suffix_filter_depth>(record.tokens, p_r + 1, r_size, s.tokens, p_s + 1, s_size, max_hd);
          if (approx_hd > max_hd) {
            continue;
          }
        }

        if constexpr (sketch_vec > 0) {
          auto& record_fp = fingerprints[record_id];
          auto max_ovlp = similarity::max_overlap(record_fp, fingerprints[cand_id]);
          // std::cerr << "vec1: " << std::endl << record_fp << std::endl;
          // std::cerr << "vec2: " << std::endl << fingerprints[cand_id] << std::endl;
          // std::cerr << "max_ovlp: " << max_ovlp << std::endl << std::endl;

          if (max_ovlp < overlap_threshold) {
            continue;
          }
        }

        int32_t p_r_until = p_r + 1;
        int32_t p_s_until = p_s + 1;
        int32_t result = util::falsify_or_compute(
          record.tokens, s.tokens, overlap_threshold, p_r_until, p_s_until, overlaps[cand_id].overlap);
        state.statistics.verifications.inc();
        state.statistics.verified_tokens.add(p_r_until - p_r + p_s_until - p_s - 2);

        if (result > 0) {
          auto sim = Similarity::sim(r_size, s_size, result);
          heap.push(cand_id, sim);

          state.statistics.successful_verifications.inc();
        }
      }
      candidates_offset = candidates.size();
      q.enqueue_next(r_size, e.lengrp_ptr, e);
    }

    state.statistics.precandidates.add(static_cast<int64_t>(candidates.size()));
    state.statistics.heap_pops.record(heap_pops);
    state.statistics.avg_verifications.record(static_cast<int64_t>(candidates.size()));
    state.statistics.final_k_similarity.record(heap.get_threshold());
    state.statistics.final_heap_size.record(static_cast<int64_t>(q.size()));
    // compare(record_id, heap);
  }

  int32_t sim_above_zero(const Record& record) {
    int32_t sum = 0;

    for (auto token : record.tokens) {
      sum += falsification_idx[token];

      if (sum >= this->k) {
        return this->k;
      }
    }
    return sum;
  }

  void compare(int32_t record_id, [[maybe_unused]] TopKHeap& heap) {
    BaselineJoin<Similarity> bj(Base::records, Base::k);
    auto bj_heap = bj.linear_scan(record_id);
    assert(heap.get_threshold() == bj_heap.get_threshold());
  }

  [[nodiscard]] static constexpr bool use_index_pos() { return pos_mode == FULL; }

private:
  indexing::inv_index<LGrouping, use_index_pos()> index;
  LGrouping& l_grouping;
  std::vector<int64_t> falsification_idx;
  indexing::dfingerprints fingerprints;
};

}  // namespace knn

#endif  // KNN_PROJECT_JOIN_HH
