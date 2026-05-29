#ifndef PARTITION_HH
#define PARTITION_HH

#include <algorithm>

#include "../../types/types.hh"
#include "../../util/set_ops.hh"
#include "../../util/tab_hash.hh"
#include "../multithreaded_join.hh"

namespace knn::partition {

struct PartitionRecord {
  std::vector<Token> remaining;
  const size_t size;

  explicit PartitionRecord(const std::vector<Token>& remaining) : remaining(remaining), size(remaining.size()) {}
};

struct SigPair {
  util::TabHash::HashVal normal{};
  std::vector<util::TabHash::HashVal> deletion;
};

struct HeapElement {
  int32_t sig;
  bool deletion = false;
  size_t cost;

  HeapElement(int32_t sig, bool deletion, size_t cost) : sig(sig), deletion(deletion), cost(cost) {}
  friend bool operator<(const HeapElement& lhs, const HeapElement& rhs) { return lhs.cost < rhs.cost; }
  friend bool operator<=(const HeapElement& lhs, const HeapElement& rhs) { return !(rhs < lhs); }
  friend bool operator>(const HeapElement& lhs, const HeapElement& rhs) { return rhs < lhs; }
  friend bool operator>=(const HeapElement& lhs, const HeapElement& rhs) { return !(lhs < rhs); }
};

struct PartThreadState {
  Candidates candidates;
  std::vector<bool> candidate_bitmap;
  statistics::JoinStatistics statistics;
  statistics::IndexStatistics index_statistics;
};

template <class Similarity, bool DELETION = false>
class PartJoin : public join::MultithreadedJoin<PartJoin<Similarity, DELETION>, PartThreadState> {
public:
  using Base = join::MultithreadedJoin<PartJoin, PartThreadState>;

public:
  PartJoin(data::SetData& data, int32_t k)
      : Base(k, data),
        partition_count(fast_ceil_log2(data.get_universe_size()) + 1),
        token_hash(data.get_universe_size()) {
    for (RecordId id = 0; id < static_cast<RecordId>(Base::records.size()); ++id) {
      auto& record = Base::records[id];
      PartitionRecord part_record(record.tokens);

      // std::cerr << "Record " << record << std::endl;

      for (int32_t sig_id = 0; sig_id < partition_count; ++sig_id) {
        auto [normal, deletion] = next_signature(part_record.remaining, sig_id);

        // std::cerr << "Sig " << sig_id << ": " << normal << std::endl;

        index[normal].push_back(id);

        if constexpr (DELETION) {
          for (auto del_hash : deletion) {
            del_index[del_hash].push_back(id);
          }
        }
      }
    }

    statistics::update([&] {
      for (auto& slist : index) {
        index_statistics.list_size.record(static_cast<int64_t>(slist.second.size()));
      }
    });
    index_statistics.set_size_bytes(calculate_index_size_bytes());
  }

  nlohmann::json get_json_statistics() const {
    nlohmann::json result;

    result["timing"] = Base::timing.to_json();
    auto final_index_statistics = Base::get_index_statistics(index_statistics, &PartThreadState::index_statistics);
    result["index"] = final_index_statistics.to_json();
    result["join"] = Base::get_join_statistics();

    return result;
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    Base::run_parallel_lookups(
      begin,
      end,
      concurrency,
      [&](PartThreadState& state) { state.candidate_bitmap.resize(Base::records.size()); },
      [&](const int32_t record_id, PartThreadState& state) { lookup(record_id, state); },
      [&](const int32_t, PartThreadState& state) {
        for (auto& cand_id : state.candidates) {
          state.candidate_bitmap[cand_id] = false;
        }
        state.candidates.clear();
      });
  }

private:
  static int32_t fast_ceil_log2(int32_t k) { return 32 - __builtin_clz(k - 1); }
  static Token fast_mod_two(Token t, int32_t k) { return t & ((1 << k) - 1); }

  SigPair next_signature(Tokens& tokens, int32_t sig_id) {
    SigPair signature{};

    ptrdiff_t left = 0;
    auto right = static_cast<ptrdiff_t>(tokens.size());

    while (left != right) {
      auto token = tokens[left];

      if (fast_mod_two(token, sig_id + 1) != 0) {
        signature.normal = token_hash.add(signature.normal, token);
        --right;
        std::swap(tokens[left], tokens[right]);
      } else {
        ++left;
      }
    }

    signature.normal = partition_hash.add(signature.normal, sig_id);

    auto it = tokens.begin() + right;
    if constexpr (DELETION) {
      auto it_copy = it;
      signature.deletion.reserve(std::distance(it_copy, tokens.end()));
      for (; it_copy != tokens.end(); ++it_copy) {
        const auto token = *it_copy;
        signature.deletion.push_back(token_hash.remove(signature.normal, token));
      }
    }
    tokens.erase(it, tokens.end());

    return signature;
  }

  double jaccard_bound(int32_t query_size, int32_t sig_id) { return 1. * query_size / (query_size + sig_id); }

  double two_sided_jaccard_bound(int32_t query_size, int32_t index_size, int32_t sig_id) {
    return 1.0 * (query_size + index_size - sig_id) / (query_size + index_size + sig_id);
  }

  size_t normal_cost(const SigPair& sig) { return index[sig.normal].size(); }

  size_t deletion_cost(const SigPair& sig) {
    size_t cost = 0;
    {
      auto it = del_index.find(sig.normal);
      if (it != del_index.end()) {
        cost += it->second.size();
      }
    }
    for (auto del_sig : sig.deletion) {
      auto it = index.find(del_sig);
      if (it != index.end()) {
        cost += it->second.size();
      }
    }
    return cost;
  }

  void handle_list(const Record& query_record,
                   const int32_t sig_id,
                   const RecordIds& list,
                   TopKHeap& result_heap,
                   PartThreadState& state) {
    auto query_size = static_cast<int32_t>(query_record.tokens.size());

    // std::cerr << "sig_id " << sig_id << " has " << list.size() << " candidates\n";

    state.statistics.precandidates.add(static_cast<int64_t>(list.size()));
    state.index_statistics.read_size.record(static_cast<int64_t>(list.size()));

    for (auto index_id : list) {
      if (!state.candidate_bitmap[index_id]) {
        state.candidate_bitmap[index_id] = true;
        state.candidates.push_back(index_id);

        auto& index_record = Base::records[index_id];
        auto index_size = static_cast<int32_t>(index_record.tokens.size());

        if (two_sided_jaccard_bound(query_size, index_size, sig_id) >= result_heap.get_threshold()) {
          auto overlap_threshold =
            Similarity::eqo(query_size, static_cast<int32_t>(index_record.tokens.size()), result_heap.get_threshold());
          int32_t result = util::falsify_or_compute(query_record.tokens, index_record.tokens, overlap_threshold);
          state.statistics.verifications.inc();

          if (result > 0) {
            auto sim = Similarity::sim(query_size, static_cast<int32_t>(index_record.tokens.size()), result);

            result_heap.push(index_id, sim);

            state.statistics.successful_verifications.inc();
          }
        }
      }
    }
  }

  void compare(int32_t record_id, const TopKHeap& heap) {
    BaselineJoin<Similarity> bj(Base::records, Base::k);
    auto bj_heap = bj.linear_scan(record_id);
    if (heap.get_threshold() < bj_heap.get_threshold()) {
      std::cerr << "Error in result: " << std::endl;
      auto heap_res = heap.get_result();
      auto bj_res = bj_heap.get_result();
      std::ranges::sort(heap_res);
      std::ranges::sort(bj_res);
      RecordIds difference(2 * Base::k);

      std::ranges::set_symmetric_difference(heap_res, bj_res, difference.begin());

      std::cerr << "Difference: ";
      for (auto d : difference) {
        std::cerr << d << ", ";
      }
      std::cerr << std::endl;
    }
  }

  void lookup(int32_t record_id, PartThreadState& state) {
    auto& query_record = Base::records[record_id];
    auto query_size = static_cast<int32_t>(query_record.tokens.size());

    PartitionRecord remainder(query_record.tokens);
    TopKHeap result_heap(Base::k);
    std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<>> sig_heap;
    std::vector<SigPair> signatures;

    signatures.reserve(partition_count);
    for (int32_t i = 0; i < partition_count; ++i) {
      auto& signature = signatures.emplace_back(next_signature(remainder.remaining, i));
      auto cost = normal_cost(signature);
      sig_heap.emplace(i, false, cost);
    }

    int32_t hamming_distance = 0;
    while (!sig_heap.empty() && result_heap.get_threshold() < jaccard_bound(query_size, hamming_distance)) {
      auto op = sig_heap.top();
      sig_heap.pop();

      if (!DELETION || !op.deletion) {
        auto& signature = signatures[op.sig];
        auto it = index.find(signature.normal);
        if (it != index.end()) {
          auto& list = it->second;
          handle_list(query_record, hamming_distance, list, result_heap, state);
        }

        if constexpr (DELETION) {
          // next deletion cost
          auto del_cost = deletion_cost(signature);
          sig_heap.emplace(op.sig, true, del_cost);
        }
      } else {
        // previous deletion signature
        auto& signature = signatures[op.sig];

        {
          // normal signature against deletion index
          auto it = del_index.find(signature.normal);
          if (it != index.end()) {
            auto& list = it->second;
            handle_list(query_record, hamming_distance, list, result_heap, state);
          }
        }

        {
          // deletion signatures against normal index
          for (auto del_sig : signature.deletion) {
            auto it = index.find(del_sig);
            if (it != index.end()) {
              auto& list = it->second;
              handle_list(query_record, hamming_distance, list, result_heap, state);
            }
          }
        }
      }

      ++hamming_distance;
    }

    state.statistics.final_heap_size.record(static_cast<int64_t>(sig_heap.size()));
    state.statistics.heap_pops.record(hamming_distance);
    state.statistics.avg_verifications.record(static_cast<int64_t>(state.candidates.size()));
    state.statistics.final_k_similarity.record(result_heap.get_threshold());
    // compare(record_id, result_heap);
  }

private:
  [[nodiscard]] size_t calculate_hash_index_size_bytes(
    const HashTable<util::TabHash::HashVal, RecordIds>& hash_index) const {
    size_t bytes = statistics::flat_hash_map_allocated_bytes(hash_index);

    for (const auto& entry : hash_index) {
      bytes += statistics::vector_payload_bytes(entry.second);
    }

    return bytes;
  }

  [[nodiscard]] size_t calculate_index_size_bytes() const {
    return calculate_hash_index_size_bytes(index) + calculate_hash_index_size_bytes(del_index);
  }

  const int32_t partition_count;

  util::TabHash token_hash;
  util::TabHash partition_hash;
  HashTable<util::TabHash::HashVal, RecordIds> index;
  HashTable<util::TabHash::HashVal, RecordIds> del_index;

  statistics::IndexStatistics index_statistics;
};

}  // namespace knn::partition

#endif  // PARTITION_HH
