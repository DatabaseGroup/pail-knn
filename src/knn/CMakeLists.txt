cmake_minimum_required(VERSION 3.13)

set(KNN_LINK_LIBS
        ssjlib
        absl::base
        absl::flat_hash_map
        absl::strings
        absl::flags
        Boost::program_options
        Boost::headers
        nlohmann_json::nlohmann_json
        puffinn::puffinn
        roaring::roaring
        TBB::tbb)

set(SOURCES
        cmdline.cc)

add_executable(knn
        ${SOURCES})
add_executable(knn_stats
        ${SOURCES})
add_executable(knn_time
        ${SOURCES})
add_executable(knn_memory
        ${SOURCES})

target_link_libraries(knn PRIVATE
        ${KNN_LINK_LIBS}
        )
target_link_libraries(knn_stats PRIVATE
        ${KNN_LINK_LIBS}
        )
target_link_libraries(knn_time PRIVATE
        ${KNN_LINK_LIBS}
        )
target_link_libraries(knn_memory PRIVATE
        ${KNN_LINK_LIBS}
        ${JEMALLOC_LIBRARY}
        )
target_include_directories(knn_memory PRIVATE ${JEMALLOC_INCLUDE_DIR})
target_compile_definitions(knn_stats PRIVATE STATISTICS TIMING)
target_compile_definitions(knn_time PRIVATE TIMING)
target_compile_definitions(knn_memory PRIVATE STATISTICS TIMING USE_JEMALLOC)
