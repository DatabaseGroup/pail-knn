cmake_minimum_required(VERSION 3.13)

add_library(ssjlib STATIC
        data/set_file_parser.cc
        ${CMAKE_CURRENT_BINARY_DIR}/util/git_sha.cc)
target_link_libraries(ssjlib
        absl::flat_hash_map
        Boost::headers)

add_subdirectory(util)

add_subdirectory(knn/)