#ifndef KNN_PROJECT_SET_DATA_HH
#define KNN_PROJECT_SET_DATA_HH

#include "../types/types.hh"

namespace data {

class SetData {
public:
  void add_record(Tokens& tokens);

  [[nodiscard]] int32_t get_universe_size() const { return universe_size; }
  Records& get_records() { return records; }

private:
  int32_t universe_size;
  Records records;
};

}


#endif //KNN_PROJECT_SET_DATA_HH
