#ifndef KNN_PROJECT_SET_FILE_PARSER_HH
#define KNN_PROJECT_SET_FILE_PARSER_HH

#include <string>

#include "../types/types.hh"
#include "set_data.hh"

namespace data {

class SetFileParser {
public:
  explicit SetFileParser(const std::string& inputFile);

public:
  SetData& get_data() { return this->set_data; }
  void parse();

private:
  SetData set_data;
  const std::string& input_file;
};

}  // namespace data

#endif  // KNN_PROJECT_SET_FILE_PARSER_HH
