#ifndef KNN_PROJECT_SET_FILE_PARSER_HH
#define KNN_PROJECT_SET_FILE_PARSER_HH

#include <string>

#include "../types/types.hh"
#include "set_data.hh"

namespace data {

class SetFileParser {
public:
  explicit SetFileParser(std::string inputFile);

public:
  SetData& get_data() {
    return this->set_data;
  }
  void parse();
private:
  SetData set_data;
  std::string input_file;
};

}


#endif //KNN_PROJECT_SET_FILE_PARSER_HH
