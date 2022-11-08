#include "set_data.hh"

namespace data {

void SetData::add_record(Tokens& tokens) {
  universe_size = std::max(universe_size, tokens.back());

  records.emplace_back(std::forward<Tokens>(tokens));
}

}
