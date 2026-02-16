#ifndef KNN_PROJECT_INV_INDEX_HH
#define KNN_PROJECT_INV_INDEX_HH

#include "../data/set_data.hh"
#include "../statistics/index_statistics.hh"
#include "../types/types.hh"
#include "../util/set_ops.hh"
#include "../util/tab_hash.hh"
#include "length_group.hh"

namespace indexing {

template <typename K, typename V>
struct skv_dense {
public:
  std::vector<K> keys;
  std::vector<V> values;

  size_t get_index(const K& key) { return util::binary_search(keys.data(), keys.size(), key); }

  [[nodiscard]] bool empty() const { return keys.empty(); }

  [[nodiscard]] size_t size() const { return values.size(); }

  template <typename... Args>
  V& emplace_back(const K key, Args&&... args) {
    keys.push_back(key);
    values.emplace_back(args...);
    return values.back();
  }
};

template <typename K, typename V>
struct skv_sparse {
public:
  skv_dense<K, size_t> keys;
  std::vector<V> values;

  [[nodiscard]] bool empty() const { return keys.empty(); }

  [[nodiscard]] size_t size() const { return values.size(); }

  typename std::vector<V>::iterator begin() { return values.begin(); }

  typename std::vector<V>::iterator end() { return values.end(); }

  template <typename... Args>
  V& emplace_back(const K key, Args&&... args) {
    keys.emplace_back(key, values.size());
    values.emplace_back(args...);
    return values.back();
  }

  template <typename... Args>
  V& emplace_sparse_back(Args&&... args) {
    values.emplace_back(args...);
    return values.back();
  }

  template <typename... Args>
  V& try_emplace(const K key, Args&&... args) {
    if (keys.empty() || keys.keys.back() != key) {
      return emplace_back(key, args...);
    } else {
      return emplace_sparse_back(args...);
    }
  }
};

struct point_hash {
  // this is a size_ptr
  int32_t size = 0;
  int32_t token = 0;
  int32_t position = 0;

  bool operator==(const point_hash& rhs) const {
    return token == rhs.token && size == rhs.size && position == rhs.position;
  }
  bool operator!=(const point_hash& rhs) const { return !(rhs == *this); }

  template <typename H>
  friend H AbslHashValue(H h, const point_hash& c) {
    return H::combine(std::move(h), c.size, c.token, c.position);
  }
};

template <class LGrouping, bool use_full_pos = false>
class inv_index {
public:
  explicit inv_index(data::SetData& data, LGrouping& lgrouping) {
    auto& records = data.get_records();

    lengrp_idx lengrp = 0;
    size_t lengrp_begin_idx = 0;
    size_t lengrp_end_idx;

    point_hash p_hash;

    while (lengrp_begin_idx < records.size()) {
      int32_t current_len_ub = lgrouping.upper_bound(lengrp);
      lengrp_end_idx = lengrp_begin_idx;
      while (lengrp_end_idx < records.size() &&
             static_cast<int32_t>(records[lengrp_end_idx].tokens.size()) <= current_len_ub) {
        ++lengrp_end_idx;
      }

      if constexpr (use_full_pos) {
        // this is a lengrp_idx
        p_hash.size = index.size();
      }
      auto& prefix_filter = index.emplace_back(lengrp);

      for (int32_t token_pos = 0; token_pos < current_len_ub; ++token_pos) {
        for (auto set_id = static_cast<int32_t>(lengrp_begin_idx); set_id < static_cast<int32_t>(lengrp_end_idx);
             ++set_id) {
          auto& record = records[set_id];

          if (static_cast<int32_t>(record.tokens.size()) <= token_pos) {
            lengrp_begin_idx = set_id;
            continue;
          }

          auto token = record.tokens[token_pos];

          auto it = prefix_filter.try_emplace(token).first;
          it->second.try_emplace(token_pos, set_id);
        }
      }

      if constexpr (use_full_pos) {
        for (auto& entry : prefix_filter) {
          auto token = entry.first;
          auto& pindex = entry.second;
          p_hash.token = token;

          for (size_t i = 0; i < pindex.keys.size(); ++i) {
            auto pos = pindex.keys.keys[i];
            p_hash.position = pos;
            auto begin = pindex.keys.values[i];
            size_t end;
            if (i + 1 < pindex.keys.size()) {
              end = pindex.keys.values[i + 1];
            } else {
              end = pindex.values.size();
            }

            point_index.try_emplace(p_hash, pindex.values.data() + begin, end - begin);
          }
        }
      }

      ++lengrp;
      lengrp_begin_idx = lengrp_end_idx;
    }

    statistics::update([&]() {
      for (auto& slist : index.values) {
        for (auto& tlist : slist) {
          statistics.list_size.record(static_cast<int64_t>(tlist.second.size()));
        }
      }
    });
  }

  void read_horizontal_filtered(const int32_t token,
                                const int32_t size_ptr,
                                const int32_t p_r,
                                const int32_t p_s_ub,
                                Candidates& candidates,
                                CandidateOverlaps& overlaps) {
    auto it = index.values[size_ptr].find(token);
    statistics.length_accesses.inc();
    statistics.token_accesses.inc();

    if (it == index.values[size_ptr].end() || it->second.empty()) {
      statistics.read_size.record(0);
      return;
    }
    auto& pindex = it->second;

    int64_t read_size = 0;
    auto pit = pindex.begin();
    for (int32_t p_idx = 0; p_idx < static_cast<int32_t>(pindex.keys.size()); ++p_idx) {
      auto p_s = pindex.keys.keys[p_idx];

      if (p_s > p_s_ub) {
        break;
      }

      auto end = ((p_idx + 1) < static_cast<int32_t>(pindex.keys.size()))
                   ? pindex.begin() +
                       static_cast<std::vector<std::vector<RecordId>>::difference_type>(pindex.keys.values[p_idx + 1])
                   : pindex.values.end();

      while (pit != end) {
        auto cand_id = *pit;
        auto& entry = overlaps[cand_id];
        if (entry.overlap == 0) {
          candidates.push_back(cand_id);
        }
        ++entry.overlap;
        entry.p_r = p_r;
        entry.p_s = p_s;
        ++pit;

        ++read_size;
      }
    }

    statistics.read_size.record(read_size);
  }

  void read_vertical_filtered(const Record& record,
                              const int32_t size_ptr,
                              const int32_t first_n_tokens,
                              const int32_t p_ub,
                              Candidates& candidates,
                              CandidateOverlaps& overlaps) {
    statistics.length_accesses.inc();

    for (int32_t p_r = 0; p_r <= first_n_tokens; ++p_r) {
      auto token = record.tokens[p_r];
      auto it = index.values[size_ptr].find(token);
      statistics.token_accesses.inc();

      if (it == index.values[size_ptr].end() || it->second.empty()) {
        statistics.read_size.record(0);
        continue;
      }
      auto& pindex = it->second;

      int64_t read_size = 0;
      auto pit = pindex.begin();
      for (int32_t p_idx = 0; p_idx < static_cast<int32_t>(pindex.keys.size()); ++p_idx) {
        auto p_s = pindex.keys.keys[p_idx];

        if (p_s > p_ub) {
          break;
        }
        auto end = ((p_idx + 1) < static_cast<int32_t>(pindex.keys.size()))
                     ? pindex.begin() +
                         static_cast<std::vector<std::vector<RecordId>>::difference_type>(pindex.keys.values[p_idx + 1])
                     : pindex.values.end();

        while (pit != end) {
          auto cand_id = *pit;
          auto& entry = overlaps[cand_id];
          if (entry.overlap == 0) {
            candidates.push_back(cand_id);
          }
          ++entry.overlap;
          entry.p_r = p_r;
          entry.p_s = p_s;
          ++pit;

          ++read_size;
        }
      }
      statistics.read_size.record(read_size);
    }
  }

  // p_r is exclusive bound!
  void read_vertical(const Record& record,
                     const int32_t size_ptr,
                     const int32_t p_r,
                     const int32_t p_s,
                     Candidates& candidates,
                     CandidateOverlaps& overlaps) {
    point_hash p_hash{size_ptr, 0, p_s};

    for (int32_t i = 0; i < p_r; ++i) {
      auto token = record.tokens[i];
      p_hash.token = token;

      auto it = point_index.find(p_hash);

      if (it == point_index.end()) {
        statistics.read_size.record(0);
        continue;
      }

      auto& list = it->second;

      for (auto cand_id : list) {
        auto& entry = overlaps[cand_id];
        if (entry.overlap == 0) {
          candidates.push_back(cand_id);
        }
        ++entry.overlap;
        entry.p_r = i;
        entry.p_s = p_s;
      }
      statistics.read_size.record(static_cast<int64_t>(list.size()));
    }
  }

  void read_cross(const Record& record,
                  const int32_t size_ptr,
                  const int32_t p_r,
                  const int32_t p_s,
                  Candidates& candidates,
                  CandidateOverlaps& overlaps) {
    auto token = record.tokens[p_r];
    read_horizontal_filtered(token, size_ptr, p_r, p_s, candidates, overlaps);
    read_vertical(record, size_ptr, p_r, p_s, candidates, overlaps);
    if (p_s > 0) {
      read_vertical(record, size_ptr, p_r, p_s - 1, candidates, overlaps);
    }
  }

  [[nodiscard]] const std::vector<RecordSize>& get_sorted_sizes() const { return index.keys; }

  [[nodiscard]] statistics::PrefixStatistics& get_statistics() { return this->statistics; }

private:
  skv_dense<RecordSize, HashTable<Token, skv_sparse<TokenPosition, RecordId>>> index;
  HashTable<point_hash, absl::Span<const int32_t>> point_index;
  statistics::PrefixStatistics statistics;
};

}  // namespace indexing

#endif  // KNN_PROJECT_INV_INDEX_HH
