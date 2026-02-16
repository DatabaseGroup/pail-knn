#ifndef KNN_PROJECT_SET_DATA_HH
#define KNN_PROJECT_SET_DATA_HH

#include "../types/types.hh"

namespace data {

class SetData {
public:
  SetData() : universe_size(0), non_unique_universe(0) {}

  void add_record(Tokens& tokens) {
    universe_size = std::max(universe_size, tokens.back());

    records.emplace_back(std::forward<Tokens>(tokens));
  }

  [[nodiscard]] int32_t get_universe_size() const { return universe_size; }
  // Records& get_records() { return records; }
  [[nodiscard]] const Records& get_records() const { return records; }

  [[nodiscard]] std::vector<int32_t> count_frequency() const {
    std::vector<int32_t> frequencies;
    frequencies.resize(get_universe_size() + 1);

    for (auto& record : records) {
      for (auto token : record.tokens) {
        ++frequencies[token];
      }
    }

    return frequencies;
  }

  [[nodiscard]] int32_t get_non_unique_universe_size() {
    if (non_unique_universe == 0) {
      std::vector mask(get_universe_size() + 1, false);
      for (auto& record : records) {
        for (auto token : record.tokens) {
          if (mask[token] == false) {
            mask[token] = true;
            ++non_unique_universe;
          }
        }
      }
    }
    return non_unique_universe;
  }

  [[nodiscard]] int64_t max_candidate_pairs() const {
    auto freqs = count_frequency();
    int64_t sum = 0;
    std::ranges::for_each(freqs, [&](auto& val) { sum += val * val; });
    return sum;
  }

  [[nodiscard]] int64_t token_count() const {
    int64_t sum = 0;
    std::ranges::for_each(records, [&](auto& val) { sum += static_cast<int64_t>(val.tokens.size()); });
    return sum;
  }

  double avg_set_size() const {
    return static_cast<double>(token_count()) / static_cast<double>(records.size());
  }

private:
  int32_t universe_size;
  int32_t non_unique_universe;
  Records records;
};

}  // namespace data

#endif  // KNN_PROJECT_SET_DATA_HH
