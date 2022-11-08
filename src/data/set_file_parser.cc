#include "set_file_parser.hh"

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace data {

SetFileParser::SetFileParser(std::string input_file) : input_file(input_file) {}

void SetFileParser::parse() {
  std::ifstream input_s(input_file, std::ios_base::in);

  if (input_s.is_open()) {
    std::string line;
    while (std::getline(input_s, line)) {
      std::stringstream line_s(line);

      std::vector<int32_t> tokens{std::istream_iterator<int32_t>(line_s),
                                  std::istream_iterator<int32_t>()};

      set_data.add_record(tokens);
    }
  } else {
    std::cerr << "Error opening file '" << input_file << "'" << std::endl;
  }
}

}
