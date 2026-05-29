#ifndef KNN_PROJECT_COMPAT_OMP_H
#define KNN_PROJECT_COMPAT_OMP_H

// Compatibility shim for PUFFINN, forces sequential computation

inline int omp_get_max_threads() {
  return 1;
}

inline int omp_get_thread_num() {
  return 0;
}

inline int omp_get_num_threads() {
  return 1;
}

inline int omp_get_num_procs() {
  return 1;
}

inline int omp_in_parallel() {
  return 0;
}

inline void omp_set_num_threads(int) {}

#endif  // KNN_PROJECT_COMPAT_OMP_H
