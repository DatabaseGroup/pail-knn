cmake_minimum_required(VERSION 3.0)

add_executable(minhash
        cmdline.cc)
target_link_libraries(minhash
        ssjlib
        boost_program_options)