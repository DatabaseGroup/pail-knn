cmake_minimum_required(VERSION 3.13)
project(knn_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)
include(GetGitRevisionDescription)

add_compile_options(-Wall -Wextra)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT error)
if (LTO_SUPPORTED)
    message(STATUS "Using Link-Time Optimization for release-builds only")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    # set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif ()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    message(STATUS "Building for native architecture (binary might be non-portable)")
    add_compile_options(-march=native)
endif ()

# used to find git hashes to include in source code
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/util/git_sha.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/src/util/git_sha.cc" @ONLY)

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# - # - # - # - # - External Libraries  # - # - # - # - # - # - # - #
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
find_package(absl CONFIG REQUIRED)
find_package(Boost REQUIRED OPTIONAL_COMPONENTS program_options)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src/)
