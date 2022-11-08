#include <cinttypes>
#include "../data/set_data.hh"
#include "../data/set_file_parser.hh"
#include "minhash_fast_kheap.hh"

int32_t main(int32_t argc, char** argv) {
  data::SetFileParser sfp("set.txt");
  sfp.parse();

  auto& dataset = sfp.get_data();
}
