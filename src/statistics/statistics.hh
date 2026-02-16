#ifndef KNN_PROJECT_STATISTICS_HH
#define KNN_PROJECT_STATISTICS_HH

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <sstream>

namespace statistics {
// The MIT License (MIT)
// Copyright (c) 2018 Willi Mann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef STATISTICS
struct RealIncreaser {
  template <class T>
  static void inc([[maybe_unused]] T) {
  }

  template <class T>
  static void add([[maybe_unused]] T, [[maybe_unused]] T) {
  }

  template <class T>
  static void set([[maybe_unused]] T& val, [[maybe_unused]] T a) {
  }

  template <class F>
  static void update([[maybe_unused]] const F& update_function) {
  }
};
#else
struct RealIncreaser {
  template <class T>
  static void inc(T& val) {
    val += 1;
  }

  template <class T>
  static void add(T& val, T a) {
    val += a;
  }

  template <class T>
  static void set(T& val, T a) {
    val = a;
  }

  template <class F>
  static void update(const F& update_function) {
    update_function();
  }
};
#endif

template <typename Increaser = RealIncreaser>
struct CountItem {
  int64_t value;
  void inc() { Increaser::inc(value); }
  void add(int64_t a) { Increaser::add(value, a); }
  void add_to_json(const std::string& name, nlohmann::json& json) const { json[name] = value; }
  void reset() { this->value = 0; }

  CountItem() : value(0) {
  }
};

template <typename Increaser = RealIncreaser>
struct AvgItem {
  int64_t sum;
  int64_t count;
  int64_t min;
  int64_t max;

  void record(int64_t a) {
    Increaser::inc(count);
    Increaser::add(sum, a);
    Increaser::set(min, std::min(min, a));
    Increaser::set(max, std::max(max, a));
  }

  [[nodiscard]] double avg() const { return count != 0 ? static_cast<double>(sum) / static_cast<double>(count) : 0; }

  void add_to_json(const std::string& name, nlohmann::json& json) const {
    json[name] = nullptr;
    json[name]["sum"] = sum;
    json[name]["count"] = count;
    json[name]["min"] = min;
    json[name]["max"] = max;
    json[name]["avg"] = avg();
  }

  void reset() {
    this->sum = 0;
    this->count = 0;
    this->min = std::numeric_limits<int64_t>::max();
    this->max = std::numeric_limits<int64_t>::min();
  }

  AvgItem() : sum(0), count(0), min(std::numeric_limits<int64_t>::max()), max(std::numeric_limits<int64_t>::min()) {
  }
};

template <typename Increaser = RealIncreaser>
struct AvgFloatItem {
  double sum;
  int64_t count;
  double min;
  double max;

  void record(double a) {
    Increaser::inc(count);
    Increaser::add(sum, a);
    Increaser::set(min, std::min(min, a));
    Increaser::set(max, std::max(max, a));
  }

  [[nodiscard]] double avg() const { return count != 0 ? sum / static_cast<double>(count) : 0; }

  void add_to_json(const std::string& name, nlohmann::json& json) const {
    json[name] = nullptr;
    json[name]["sum"] = sum;
    json[name]["count"] = count;
    json[name]["min"] = min;
    json[name]["max"] = max;
    json[name]["avg"] = avg();
  }

  void reset() {
    this->sum = 0;
    this->count = 0;
    this->min = std::numeric_limits<double>::max();
    this->max = std::numeric_limits<double>::min();
  }

  AvgFloatItem() : sum(0), count(0), min(std::numeric_limits<double>::max()), max(std::numeric_limits<double>::min()) {
  }
};

template <class F, class Increaser = RealIncreaser>
static void update(const F& update_function) {
  Increaser::update(update_function);
}

template <int32_t DIM, typename Increaser = RealIncreaser>
struct HistItem {
  std::unordered_map<int64_t, std::pair<std::array<int64_t, DIM>, std::string>> hist;

  void record(int64_t key, const std::array<int64_t, DIM>& values, const std::string& label = "") {
    Increaser::update([&] {
      auto& e = hist[key];
      for (size_t i = 0; i < DIM; ++i) {
        e.first[i] += values[i];
      }
      e.second = label;
    });
  }

  void reset() {
    Increaser::update([&] { hist.clear(); });
  }

  void add_to_json(const std::string& name, nlohmann::json& json) const {
    json[name] = nullptr;

    std::vector<std::pair<int64_t, std::pair<std::array<int64_t, DIM>, std::string>>> entries;
    entries.reserve(hist.size());
    for (auto& e : hist) {
      entries.emplace_back(e);
    }
    std::ranges::sort(entries, std::less{}, [](auto& o) { return o.first; });

    std::vector<nlohmann::json> result;
    for (auto& e : entries) {
      auto& v = result.emplace_back();
      v.emplace_back(e.first);
      for (auto k : e.second.first) {
        v.emplace_back(k);
      }
      v.emplace_back(e.second.second);
    }

    json[name] = result;
  }
};
} // namespace statistics

#endif  // KNN_PROJECT_STATISTICS_HH