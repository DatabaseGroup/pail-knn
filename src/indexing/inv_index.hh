#ifndef KNN_PROJECT_INV_INDEX_HH
#define KNN_PROJECT_INV_INDEX_HH

#include <absl/types/span.h>

#include "../data/set_data.hh"
#include "../statistics/index_statistics.hh"
#include "../types/types.hh"
#include "length_group.hh"

namespace indexing {

struct posting_segment {
  TokenPosition position;
  size_t begin;
};

struct position_postings {
  std::vector<posting_segment> segments;
  RecordIds record_ids;

  void add(const TokenPosition position, const RecordId record_id) {
    if (segments.empty() || segments.back().position != position) {
      segments.push_back({position, record_ids.size()});
    }

    record_ids.push_back(record_id);
  }

  [[nodiscard]] bool empty() const { return record_ids.empty(); }

  [[nodiscard]] size_t size() const { return record_ids.size(); }

  [[nodiscard]] size_t segment_count() const { return segments.size(); }

  [[nodiscard]] TokenPosition segment_position(const size_t i) const { return segments[i].position; }

  [[nodiscard]] absl::Span<const RecordId> segment_records(const size_t i) const {
    const auto begin = segments[i].begin;
    const auto end = i + 1 < segments.size() ? segments[i + 1].begin : record_ids.size();
    return absl::MakeConstSpan(record_ids.data() + begin, end - begin);
  }
};

using group_index = HashTable<Token, position_postings>;

struct point_key {
  lengrp_idx group = 0;
  Token token = 0;
  TokenPosition position = 0;

  bool operator==(const point_key& rhs) const {
    return token == rhs.token && group == rhs.group && position == rhs.position;
  }
  bool operator!=(const point_key& rhs) const { return !(rhs == *this); }

  template <typename H>
  friend H AbslHashValue(H h, const point_key& c) {
    return H::combine(std::move(h), c.group, c.token, c.position);
  }
};

template <class LGrouping, bool use_full_pos = false>
class inv_index {
public:
  explicit inv_index(data::SetData& data, const LGrouping& lgrouping) {
    auto& records = data.get_records();

    lengrp_idx lengrp = 0;
    size_t lengrp_begin_idx = 0;
    size_t lengrp_end_idx;

    point_key p_key;

    while (lengrp_begin_idx < records.size()) {
      int32_t current_len_ub = lgrouping.upper_bound(lengrp);
      lengrp_end_idx = lengrp_begin_idx;
      while (lengrp_end_idx < records.size() &&
             static_cast<int32_t>(records[lengrp_end_idx].tokens.size()) <= current_len_ub) {
        ++lengrp_end_idx;
      }

      if constexpr (use_full_pos) {
        // this is a lengrp_idx
        p_key.group = static_cast<lengrp_idx>(index.size());
      }
      auto& prefix_filter = index.emplace_back();

      for (int32_t token_pos = 0; token_pos < current_len_ub; ++token_pos) {
        for (auto set_id = static_cast<int32_t>(lengrp_begin_idx); set_id < static_cast<int32_t>(lengrp_end_idx);
             ++set_id) {
          auto& record = records[set_id];

          if (static_cast<int32_t>(record.tokens.size()) <= token_pos) {
            lengrp_begin_idx = set_id;
            continue;
          }

          auto token = record.tokens[token_pos];

          prefix_filter[token].add(token_pos, set_id);
        }
      }

      if constexpr (use_full_pos) {
        for (auto& entry : prefix_filter) {
          auto token = entry.first;
          auto& pindex = entry.second;
          p_key.token = token;

          for (size_t i = 0; i < pindex.segment_count(); ++i) {
            p_key.position = pindex.segment_position(i);
            point_index.try_emplace(p_key, pindex.segment_records(i));
          }
        }
      }

      ++lengrp;
      lengrp_begin_idx = lengrp_end_idx;
    }

    statistics::update([&]() {
      for (auto& slist : index) {
        for (auto& tlist : slist) {
          statistics.list_size.record(static_cast<int64_t>(tlist.second.size()));
        }
      }
    });
    statistics.set_size_bytes(calculate_size_bytes());
  }

  void read_horizontal_filtered(const int32_t token,
                                const int32_t size_ptr,
                                const int32_t p_r,
                                const int32_t p_s_ub,
                                Candidates& candidates,
                                CandidateOverlaps& overlaps,
                                statistics::PrefixStatistics& query_statistics) const {
    const auto& token_index = index[size_ptr];
    auto it = token_index.find(token);
    query_statistics.length_accesses.inc();
    query_statistics.token_accesses.inc();

    if (it == token_index.end() || it->second.empty()) {
      query_statistics.read_size.record(0);
      return;
    }
    const auto& pindex = it->second;

    const auto read_size = scan_segments(pindex, p_s_ub, [&](const RecordId cand_id, const TokenPosition p_s) {
      auto& entry = overlaps[cand_id];
      if (entry.overlap == 0) {
        candidates.push_back(cand_id);
      }
      ++entry.overlap;
      entry.p_r = p_r;
      entry.p_s = p_s;
    });

    query_statistics.read_size.record(read_size);
  }

  void read_vertical_filtered(const Record& record,
                              const int32_t size_ptr,
                              const int32_t first_n_tokens,
                              const int32_t p_ub,
                              Candidates& candidates,
                              CandidateOverlaps& overlaps,
                              statistics::PrefixStatistics& query_statistics) const {
    query_statistics.length_accesses.inc();
    const auto& token_index = index[size_ptr];

    for (int32_t p_r = 0; p_r <= first_n_tokens; ++p_r) {
      auto token = record.tokens[p_r];
      auto it = token_index.find(token);
      query_statistics.token_accesses.inc();

      if (it == token_index.end() || it->second.empty()) {
        query_statistics.read_size.record(0);
        continue;
      }
      const auto& pindex = it->second;

      const auto read_size = scan_segments(pindex, p_ub, [&](const RecordId cand_id, const TokenPosition p_s) {
        auto& entry = overlaps[cand_id];
        if (entry.overlap == 0) {
          candidates.push_back(cand_id);
        }
        ++entry.overlap;
        entry.p_r = p_r;
        entry.p_s = p_s;
      });
      query_statistics.read_size.record(read_size);
    }
  }

  // p_r is exclusive bound!
  void read_vertical(const Record& record,
                     const int32_t size_ptr,
                     const int32_t p_r,
                     const int32_t p_s,
                     Candidates& candidates,
                     CandidateOverlaps& overlaps,
                     statistics::PrefixStatistics& query_statistics) const {
    point_key p_key{size_ptr, 0, p_s};

    for (int32_t i = 0; i < p_r; ++i) {
      auto token = record.tokens[i];
      p_key.token = token;

      auto it = point_index.find(p_key);

      if (it == point_index.end()) {
        query_statistics.read_size.record(0);
        continue;
      }

      const auto& list = it->second;

      for (auto cand_id : list) {
        auto& entry = overlaps[cand_id];
        if (entry.overlap == 0) {
          candidates.push_back(cand_id);
        }
        ++entry.overlap;
        entry.p_r = i;
        entry.p_s = p_s;
      }
      query_statistics.read_size.record(static_cast<int64_t>(list.size()));
    }
  }

  void read_cross(const Record& record,
                  const int32_t size_ptr,
                  const int32_t p_r,
                  const int32_t p_s,
                  Candidates& candidates,
                  CandidateOverlaps& overlaps,
                  statistics::PrefixStatistics& query_statistics) const {
    auto token = record.tokens[p_r];
    read_horizontal_filtered(token, size_ptr, p_r, p_s, candidates, overlaps, query_statistics);
    read_vertical(record, size_ptr, p_r, p_s, candidates, overlaps, query_statistics);
    if (p_s > 0) {
      read_vertical(record, size_ptr, p_r, p_s - 1, candidates, overlaps, query_statistics);
    }
  }

  [[nodiscard]] statistics::PrefixStatistics& get_statistics() { return this->statistics; }
  [[nodiscard]] const statistics::PrefixStatistics& get_statistics() const { return this->statistics; }

private:
  [[nodiscard]] size_t calculate_size_bytes() const {
    size_t bytes = statistics::vector_allocated_bytes(index);

    for (const auto& group : index) {
      bytes += statistics::flat_hash_map_payload_bytes(group);
      for (const auto& entry : group) {
        const auto& postings = entry.second;
        bytes += statistics::vector_payload_bytes(postings.segments);
        bytes += statistics::vector_payload_bytes(postings.record_ids);
      }
    }

    bytes += statistics::flat_hash_map_allocated_bytes(point_index);

    return bytes;
  }

  template <typename Callback>
  static int64_t scan_segments(const position_postings& pindex, const TokenPosition p_s_ub, Callback&& callback) {
    int64_t read_size = 0;
    for (size_t i = 0; i < pindex.segment_count(); ++i) {
      const auto p_s = pindex.segment_position(i);

      if (p_s > p_s_ub) {
        break;
      }

      for (const auto cand_id : pindex.segment_records(i)) {
        callback(cand_id, p_s);
        ++read_size;
      }
    }

    return read_size;
  }

  std::vector<group_index> index;
  HashTable<point_key, absl::Span<const RecordId>> point_index;
  statistics::PrefixStatistics statistics;
};

}  // namespace indexing

#endif  // KNN_PROJECT_INV_INDEX_HH
