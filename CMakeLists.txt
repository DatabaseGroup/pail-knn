cmake_minimum_required(VERSION 3.18)
project(knn_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)
include(FetchContent)
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
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_path(JEMALLOC_INCLUDE_DIR jemalloc/jemalloc.h REQUIRED)
find_library(JEMALLOC_LIBRARY jemalloc REQUIRED)

# - # - # - # - # - PUFFINN Subsection - # - # - # - # - # - # - # - #

FetchContent_Declare(
        puffinn
        GIT_REPOSITORY https://github.com/puffinn/puffinn.git
        GIT_TAG fd86b0d6591450dcaf32aa8f7bd988f7301adb1a
        SOURCE_SUBDIR include)
FetchContent_MakeAvailable(puffinn)
FetchContent_GetProperties(puffinn)

add_library(puffinn_puffinn INTERFACE)
add_library(puffinn::puffinn ALIAS puffinn_puffinn)
set(PUFFINN_PATCHED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/puffinn_patched_include")
file(REMOVE_RECURSE "${PUFFINN_PATCHED_INCLUDE_DIR}")
file(COPY "${puffinn_SOURCE_DIR}/include/" DESTINATION "${PUFFINN_PATCHED_INCLUDE_DIR}")

# We slightly patch some PUFFINN files:
# 1. filterer.hpp includes <immintrin.h> (but does not really use it), so this does not compile on ARM
#    We remove it as it is unused anyway.
set(PUFFINN_FILTERER_HPP "${PUFFINN_PATCHED_INCLUDE_DIR}/puffinn/filterer.hpp")
file(READ "${PUFFINN_FILTERER_HPP}" PUFFINN_FILTERER_CONTENTS)
set(PUFFINN_FILTERER_IMMINTRIN_INCLUDE "#include <immintrin.h>\n")
string(FIND "${PUFFINN_FILTERER_CONTENTS}" "${PUFFINN_FILTERER_IMMINTRIN_INCLUDE}" PUFFINN_FILTERER_IMMINTRIN_POS)
string(REPLACE "${PUFFINN_FILTERER_IMMINTRIN_INCLUDE}" "" PUFFINN_FILTERER_CONTENTS "${PUFFINN_FILTERER_CONTENTS}")
file(WRITE "${PUFFINN_FILTERER_HPP}" "${PUFFINN_FILTERER_CONTENTS}")

# 2. We want access to the statistics of PUFFINN for the statistics build. This is disabled by a constant variable.
#    We patch the variable to depend on the value of the same macro used for the remaining project.
find_program(PATCH_EXECUTABLE patch)
if (NOT PATCH_EXECUTABLE)
    message(FATAL_ERROR "Could not find the patch executable")
endif ()
execute_process(
        COMMAND "${PATCH_EXECUTABLE}" -p0 -i "${CMAKE_CURRENT_SOURCE_DIR}/cmake/puffinn/performance.hpp.patch"
        WORKING_DIRECTORY "${PUFFINN_PATCHED_INCLUDE_DIR}"
        RESULT_VARIABLE PUFFINN_PERFORMANCE_PATCH_RESULT
        OUTPUT_VARIABLE PUFFINN_PERFORMANCE_PATCH_OUTPUT
        ERROR_VARIABLE PUFFINN_PERFORMANCE_PATCH_ERROR)
if (NOT PUFFINN_PERFORMANCE_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Failed to patch PUFFINN performance.hpp\n"
            "${PUFFINN_PERFORMANCE_PATCH_OUTPUT}"
            "${PUFFINN_PERFORMANCE_PATCH_ERROR}")
endif ()
target_include_directories(puffinn_puffinn SYSTEM INTERFACE "${PUFFINN_PATCHED_INCLUDE_DIR}")
target_include_directories(puffinn_puffinn INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/compat")

# - # - # - # - # - PUFFINN Subsection End - # - # - # - # - # - # - #

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src/)
