cmake_minimum_required(VERSION 3.0)

add_library(ssjlib STATIC
        data/set_data.cc
        data/set_file_parser.cc)

add_subdirectory(minhash/)