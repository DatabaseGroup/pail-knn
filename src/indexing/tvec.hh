#ifndef KNN_PROJECT_TVEC_HH
#define KNN_PROJECT_TVEC_HH

#include "../types/types.hh"

namespace indexing {

template <typename T, typename Alloc = std::allocator<T>>
class tvec {
private:
  static constexpr size_t tcap = (sizeof(std::vector<T, Alloc>) - sizeof(int16_t)) / sizeof(T);

  union vec_or_invec {
    std::vector<T, Alloc> vec;
    struct {
      uint32_t size;
      T dat[tcap];
    } invec;

    vec_or_invec() {
      this->invec.size = 0;
      std::memset(this->invec.dat, 0, tcap);
    }

    ~vec_or_invec() {
      if (uses_inline()) {
        for (size_t i = 0; i < this->invec.size; ++i) {
          std::destroy_at<T>(&this->invec.dat[i]);
        }
      } else {
        std::destroy_at<std::vector<T, Alloc>>(&this->vec);
      }
    }

    bool uses_inline() {
      // there are no memory addresses this small for 64 bit machines
      return this->invec.size < tcap;
    }
  } data;

public:
  tvec() = default;

  ~tvec() = default;

  size_t size() {
    if (this->data.uses_inline()) {
      return this->data.invec.size;
    } else {
      return this->data.vec.size();
    }
  }

  void push_back(const T& x) {
    if (this->data.uses_inline()) {
      if (this->data.invec.size < tcap - 1) {
        this->data.invec.dat[this->data.invec.size] = x;
        ++this->data.invec.size;
      } else {
        std::vector<T, Alloc> new_vector;
        new_vector.reserve(tcap * 2);

        for (size_t i = 0; i < this->data.invec.size; ++i) {
          new_vector.push_back(std::move(this->data.invec.dat[i]));
        }

        construct_at(&this->data.vec, new_vector);
      }
    } else {
      this->data.vec.push_back(x);
    }
  }

  T& operator[](size_t i) {
    if (this->data.uses_inline()) {
      return this->data.invec.dat[i];
    } else {
      return this->data.vec[i];
    }
  }

private:
  template <class U, class... Args>
  constexpr U* construct_at(U* p, Args&&... args) {
    return ::new (const_cast<void*>(static_cast<const volatile void*>(p))) U(std::forward<Args>(args)...);
  }
};

}  // namespace indexing

#endif  // KNN_PROJECT_TVEC_HH
