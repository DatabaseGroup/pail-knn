#ifndef KNN_PROJECT_LES3_HH
#define KNN_PROJECT_LES3_HH

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../statistics/index_statistics.hh"
#include "../../statistics/join_statistics.hh"
#include "../../types/types.hh"
#include "../multithreaded_join.hh"
#include "tgm_c.hh"

namespace knn::les3 {

using std::multiset;
using std::pair;
using std::priority_queue;
using std::vector;

struct RecordIdPair {
  RecordId id;
  Record record;

  RecordIdPair(const RecordId id, Record record) : id(id), record(std::move(record)) {}
};

struct LES3 {
  int32_t num_groups = 0;
  TGM_C tgm;
  int32_t total_num_sets = 0;
  const Records& records;
  vector<vector<int32_t>> groups;
  vector<vector<RecordIdPair>> grouped_database;

  LES3(const Records& records, const std::filesystem::path& path_to_groups) : records(records) {
    total_num_sets = static_cast<int32_t>(records.size());
    preprocess(path_to_groups);
    tgm.construct(records, groups);
  }

  void preprocess(const std::filesystem::path& path_to_groups) {
    read_groups(path_to_groups);
    construct_grouped_database();
    num_groups = static_cast<int32_t>(groups.size());
  }

  [[nodiscard]] int32_t findKNN(const std::vector<int32_t>& query_set, int32_t k) const;
  int32_t findKNN(const std::vector<int32_t>& query_set,
                  int32_t k,
                  statistics::JoinStatistics& statistics,
                  statistics::IndexStatistics& index_statistics) const;

  [[nodiscard]] int64_t get_size_in_MB() const { return tgm.get_size(); }

  [[nodiscard]] std::size_t size_bytes() const { return tgm.size_bytes(); }

private:
  void read_groups(const std::filesystem::path& path_to_groups) {
    std::ifstream input(path_to_groups);
    if (!input.is_open()) {
      throw std::runtime_error("Error opening LES3 group file '" + path_to_groups.string() + "'");
    }

    groups.clear();
    std::vector seen(records.size(), false);
    std::size_t seen_count = 0;

    std::string line;
    int64_t line_no = 0;
    while (std::getline(input, line)) {
      ++line_no;
      std::istringstream line_stream(line);
      std::string raw_id;
      vector<int32_t> current_group;
      while (line_stream >> raw_id) {
        std::size_t parsed = 0;
        const int64_t parsed_id = std::stoll(raw_id, &parsed);
        if (parsed != raw_id.size()) {
          throw std::runtime_error("Invalid record id in LES3 group file at line " + std::to_string(line_no));
        }
        if (parsed_id < 0 || parsed_id >= static_cast<int64_t>(records.size())) {
          throw std::runtime_error("Out-of-range record id in LES3 group file at line " + std::to_string(line_no));
        }

        const auto set_id = static_cast<int32_t>(parsed_id);
        if (seen[set_id]) {
          throw std::runtime_error("Duplicate record id in LES3 group file: " + std::to_string(set_id));
        }
        seen[set_id] = true;
        ++seen_count;
        current_group.push_back(set_id);
      }
      groups.push_back(std::move(current_group));
    }

    if (groups.empty() && !records.empty()) {
      throw std::runtime_error("LES3 group file does not contain any groups");
    }
    if (seen_count != records.size()) {
      for (std::size_t set_id = 0; set_id < seen.size(); ++set_id) {
        if (!seen[set_id]) {
          throw std::runtime_error("Missing record id in LES3 group file: " + std::to_string(set_id));
        }
      }
    }
  }

  void construct_grouped_database() {
    grouped_database.clear();
    grouped_database.reserve(groups.size());
    for (const auto& group : groups) {
      vector<RecordIdPair> this_group;
      this_group.reserve(group.size());
      for (const int32_t set_id : group) {
        this_group.emplace_back(set_id, records[set_id]);
      }
      grouped_database.push_back(std::move(this_group));
    }
  }
};

inline int32_t LES3::findKNN(const std::vector<int32_t>& query_set, int32_t k) const {
  int32_t check_counts = 0;
  priority_queue<pair<double, int32_t>> candidate_groups;
  for (int32_t j = 0; j < num_groups; j++) {
    double ub = tgm.getUB(query_set, j);
    candidate_groups.emplace(ub, j);
  }

  TopKHeap result_heap{k};
  while (!candidate_groups.empty()) {
    if (result_heap.get_threshold() >= candidate_groups.top().first) {
      break;
    }
    for (auto& candidate_set : grouped_database[candidate_groups.top().second]) {
      check_counts++;

      auto candidate_size = static_cast<int32_t>(candidate_set.record.tokens.size());

      auto eqo = similarity::JaccardSimilarity::eqo(
        static_cast<int32_t>(query_set.size()), candidate_size, candidate_groups.top().second);
      int32_t ovlp = util::falsify_or_compute(query_set, candidate_set.record.tokens, eqo);
      if (ovlp > 0) {
        auto sim = similarity::JaccardSimilarity::sim(static_cast<int32_t>(query_set.size()), candidate_size, ovlp);
        result_heap.push(candidate_set.id, sim);
      }
    }
    candidate_groups.pop();
  }
  // cout<<check_counts<<endl;
  return check_counts;
}

inline int32_t LES3::findKNN(const std::vector<int32_t>& query_set,
                             int32_t k,
                             statistics::JoinStatistics& statistics,
                             statistics::IndexStatistics& index_statistics) const {
  int32_t check_counts = 0;
  int64_t successful_verifications = 0;
  int64_t group_pops = 0;
  priority_queue<pair<double, int32_t>> candidate_groups;
  for (int32_t j = 0; j < num_groups; j++) {
    double ub = tgm.getUB(query_set, j);
    candidate_groups.emplace(ub, j);
  }

  TopKHeap result_heap{k};
  while (!candidate_groups.empty()) {
    if (result_heap.get_threshold() >= candidate_groups.top().first) {
      break;
    }

    auto& group = grouped_database[candidate_groups.top().second];
    index_statistics.read_size.record(static_cast<int64_t>(group.size()));
    for (auto& candidate_set : grouped_database[candidate_groups.top().second]) {
      check_counts++;

      auto candidate_size = static_cast<int32_t>(candidate_set.record.tokens.size());

      auto eqo = similarity::JaccardSimilarity::eqo(
        static_cast<int32_t>(query_set.size()), candidate_size, result_heap.get_threshold());
      int32_t ovlp = util::falsify_or_compute(query_set, candidate_set.record.tokens, eqo);
      if (ovlp >= 0) {
        successful_verifications++;
        auto sim = similarity::JaccardSimilarity::sim(static_cast<int32_t>(query_set.size()), candidate_size, ovlp);
        result_heap.push(candidate_set.id, sim);
      }
    }
    candidate_groups.pop();
    group_pops++;
  }

  statistics.precandidates.add(check_counts);
  statistics.verifications.add(check_counts);
  statistics.avg_verifications.record(check_counts);
  statistics.successful_verifications.add(successful_verifications);
  statistics.heap_pops.record(group_pops);
  statistics.final_heap_size.record(static_cast<int64_t>(candidate_groups.size()));
  statistics.final_k_similarity.record(result_heap.get_threshold());

  return check_counts;
}

struct LES3ThreadState {
  statistics::JoinStatistics statistics;
  statistics::IndexStatistics index_statistics;
};

class LES3Join : public join::MultithreadedJoin<LES3Join, LES3ThreadState> {
public:
  using Base = MultithreadedJoin<LES3Join, LES3ThreadState>;

  LES3Join(const data::SetData& data, const int32_t k, const std::filesystem::path& path_to_groups)
      : Base(k, data), les3(data.get_records(), path_to_groups) {
    if (k <= 0) {
      throw std::invalid_argument("LES3 requires k > 0");
    }

    statistics::update([&] {
      for (const auto& group : les3.groups) {
        index_statistics.list_size.record(static_cast<int64_t>(group.size()));
      }
    });
    index_statistics.set_size_bytes(les3.size_bytes());
  }

  [[nodiscard]] nlohmann::json get_json_statistics() const {
    nlohmann::json json;

    json["timing"] = timing.to_json();
    const auto final_index_statistics = get_index_statistics(index_statistics, &LES3ThreadState::index_statistics);
    json["index"] = final_index_statistics.to_json();
    json["index"]["group_count"] = les3.num_groups;
    json["join"] = get_join_statistics();

    return json;
  }

  template <class Iterator>
  void do_multiple_lookups(Iterator begin, Iterator end, const int32_t concurrency) {
    Base::run_parallel_lookups(
      begin,
      end,
      concurrency,
      [](LES3ThreadState&) {},
      [&](const int32_t record_id, LES3ThreadState& state) { lookup(record_id, state); },
      [](const int32_t, LES3ThreadState&) {});
  }

private:
  void lookup(const int32_t record_id, LES3ThreadState& state) const {
    auto& query_set = les3.records[record_id];
    les3.findKNN(query_set.tokens, k, state.statistics, state.index_statistics);
  }

private:
  LES3 les3;
  statistics::IndexStatistics index_statistics;
};

}  // namespace knn::les3

#endif  // KNN_PROJECT_LES3_HH
