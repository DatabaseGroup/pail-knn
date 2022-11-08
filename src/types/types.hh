#ifndef KNN_PROJECT_TYPES_HH
#define KNN_PROJECT_TYPES_HH

#include <utility>
#include <vector>
#include <cinttypes>

using RecordId = int32_t;
using Token = int32_t;
using Tokens = std::vector<Token>;

struct Record{
  Tokens tokens;

  explicit Record(Tokens&& tokens) : tokens(std::move(tokens)) {}
};

using Records = std::vector<Record>;

#endif  // KNN_PROJECT_TYPES_HH
