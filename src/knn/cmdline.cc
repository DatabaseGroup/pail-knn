#include <boost/program_options.hpp>
#include <filesystem>
#include <thread>

#include "../data/set_file_parser.hh"
#include "../indexing/inv_index.hh"
#include "../util/git_sha.hh"
#include "baseline.hh"
#include "baselinepp.hh"
#include "join.hh"
#include "partition/partition.hh"

struct Config {
  std::string input_file;
  std::string algorithm;
  int32_t min_batch_size = 1;
  int32_t max_batch_size = 1;
  int32_t concurrency = -1;
  std::string mode;
  int32_t sample_size = 0;
  uint64_t seed = 0;
  int32_t k = 0;
  int32_t suffix_depth = 0;
  int32_t sketch_vec = 0;
  std::string label;
  std::string length_grouping;
  bool deletion = false;
};

enum Algorithm { TOPKBASELINE, SLIM, FULL, TRANSFORMATION, PALLOC, PARTITION, BASELINE, BASELINEPP };

bool process_program_options(int argc, char** argv, Config& config) {
  namespace po = boost::program_options;

  po::options_description optdesc{"USAGE"};
  optdesc.add_options()("input-file,f", po::value(&config.input_file)->required(), "Input file")(
    "top-k,k",
    po::value(&config.k)->required(),
    "result size")(
    "algorithm,a",
    po::value(&config.algorithm)->default_value("slim"),
    "One of the following: baseline, baselinepp, topkbaseline, slim, full, transformation, partition, palloc")(
    "min-batch-size,y",
    po::value(&config.min_batch_size)->default_value(1),
    "Minimum batch size used for baseline and baselinepp")(
    "max-batch-size,z",
    po::value(&config.max_batch_size)->default_value(1),
    "Maximum batch size used for baseline and baselinepp")(
    "concurrency,c",
    po::value(&config.concurrency)->default_value(std::thread::hardware_concurrency()),
    "Level of concurrency/number of threads. -1 for all cores")(
    "sample-size,s",
    po::value(&config.sample_size)->default_value(10000),
    "Sample size for sampled join")(
    "seed,r",
    po::value(&config.seed)->default_value(0x42424242),
    "Seed for prng")(
    "mode,m",
    po::value(&config.mode)->default_value("sample"),
    "Sample or join")(
    "suffix,x",
    po::value(&config.suffix_depth)->default_value(0),
    "suffix depth for filtering")(
    "vector,v",
    po::value(&config.sketch_vec)->default_value(0),
    "Activate filtering using vectors")(
    "label,l",
    po::value(&config.label)->default_value(""),
    "Label for the runs (printed in json)")(
    "length-grouping,g",
    po::value(&config.length_grouping)->default_value("identity"),
    "Select length grouping (identity, wsqrt, usqrt, aio)")(
    "deletion,d",
    po::bool_switch(&config.deletion),
    "Enable deletion signatures for partition");

  const std::string exec_name(argv[0]);

  try {
    po::variables_map vm;
    po::command_line_parser parser(argc, argv);
    po::store(parser.options(optdesc).run(), vm);

    po::notify(vm); // update variables map
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << optdesc << "\n";
    return false;
  }

  if (config.algorithm == "topkbaseline" && config.length_grouping != "aio") {
    std::cerr << "WARN: topkbaseline only supports aio grouping! Ignoring value " << config.length_grouping
      << std::endl;
    config.length_grouping = "aio";
  }

  if (config.algorithm != "baseline" && config.algorithm != "baselinepp" && (config.min_batch_size != 1 || config.max_batch_size != 0)) {
    std::cerr << "WARN: min_batch_size and max_batch_size are only supported for baseline and baselinepp! Ignoring parameters." << std::endl;
  }

  if (config.min_batch_size > config.max_batch_size) {
    std::cerr << "WARN: min_batch_size higher than max_batch_size! Setting min_batch_size := max_batch_size;" << std::endl;
    config.min_batch_size = config.max_batch_size;
  }

  if (config.concurrency == -1) {
    config.concurrency = std::thread::hardware_concurrency();
  }

  return true;
}

nlohmann::json get_build_info() {
  nlohmann::json json;

#ifdef STATISTICS
  json["type"] = "statistics";
#else
  json["type"] = "timing";
#endif

  json["commit_sha1"] = util::GIT_SHA;

  return json;
}

nlohmann::json getISOCurrentTimestamp() {
  nlohmann::json time;
  absl::Time t1 = absl::Now();
  time["$date"] = absl::FormatTime("%Y-%m-%d%ET%H:%M:%SZ", t1, absl::UTCTimeZone());
  return time;
}

template <Algorithm algorithm>
nlohmann::json get_metadata(const Config& config, data::SetFileParser& sfp) {
  nlohmann::json json;

  json["date"] = getISOCurrentTimestamp();
  json["build"] = get_build_info();
  json["cardinality"] = sfp.get_data().get_records().size();
  json["concurrency"] = config.concurrency;
  json["dataset"] = std::filesystem::path(config.input_file).filename();
  json["k"] = config.k;
  json["mode"] = config.mode;
  if (config.mode == "sample") {
    json["sample_size"] = config.sample_size;
    json["seed"] = config.seed;
  }
  json["algorithm"] = config.algorithm;
  if constexpr (algorithm == FULL || algorithm == SLIM || algorithm == TOPKBASELINE) {
    json["size_grouping"] = config.length_grouping;
    json["suffix_depth"] = config.suffix_depth;
    json["sketch_vec"] = config.sketch_vec;
  }
  if constexpr (algorithm == BASELINE || algorithm == BASELINEPP) {
    json["min_batch_size"] = config.min_batch_size;
    json["max_batch_size"] = config.max_batch_size;
  }
  json["label"] = config.label;

  return json;
}

template <class LGrouping, int32_t suffix_depth, int32_t sketch_vec, Algorithm algorithm>
nlohmann::json execute_for_lgrouping(Config& config, data::SetFileParser& sfp, LGrouping& l_grouping) {
  nlohmann::json result;

  timing::RealTimer outer_time;

  outer_time.start();

  if (algorithm == TOPKBASELINE) {
    knn::KNNJoin<similarity::JaccardSimilarity, LGrouping, knn::PositionalMode::TOPK_BASELINE, suffix_depth, sketch_vec>
      join(sfp.get_data(), config.k, l_grouping);
    if (config.mode == "sample") {
      join.sample_and_query(config.sample_size, config.seed, config.concurrency);
    } else {
      join.join(config.concurrency);
    }
    result["statistics"] = join.get_json_statistics();
  } else if (algorithm == SLIM) {
    knn::KNNJoin<similarity::JaccardSimilarity, LGrouping, knn::PositionalMode::FILTER, suffix_depth, sketch_vec> join(
      sfp.get_data(),
      config.k,
      l_grouping);
    if (config.mode == "sample") {
      join.sample_and_query(config.sample_size, config.seed, config.concurrency);
    } else {
      join.join(config.concurrency);
    }
    result["statistics"] = join.get_json_statistics();
  } else {
    knn::KNNJoin<similarity::JaccardSimilarity, LGrouping, knn::PositionalMode::FULL, suffix_depth, sketch_vec> join(
      sfp.get_data(),
      config.k,
      l_grouping);
    if (config.mode == "sample") {
      join.sample_and_query(config.sample_size, config.seed, config.concurrency);
    } else {
      join.join(config.concurrency);
    }
    result["statistics"] = join.get_json_statistics();
  }
  outer_time.stop();

  result["meta"] = get_metadata<algorithm>(config, sfp);
  result["meta"]["outer_time"] = outer_time.get();

  return result;
}

template <int32_t suffix_depth, int32_t sketch_vec, Algorithm algorithm>
nlohmann::json execute_for_suffix(Config& config, data::SetFileParser& sfp) {
  if (config.length_grouping == "identity") {
    indexing::len_hist hist = indexing::build_length_histogram(sfp.get_data().get_records());
    indexing::ExactLengthGrouping grouping(hist);
    return execute_for_lgrouping<indexing::ExactLengthGrouping, suffix_depth, sketch_vec, algorithm>(
      config,
      sfp,
      grouping);
  } else if (config.length_grouping == "usqrt") {
    indexing::len_hist hist = indexing::build_length_histogram(sfp.get_data().get_records());
    indexing::BalancedFunctionalLengthGrouping grouping(hist,
                                                        [&]([[maybe_unused]] size_t _) {
                                                          // How much faster is scanning compared to switching between lists?
                                                          constexpr int64_t scan_bias = 2;
                                                          return static_cast<size_t>(std::max(
                                                            static_cast<int64_t>(std::round(std::sqrt(
                                                              1. * sfp.get_data().token_count() / static_cast<double>(
                                                                scan_bias * sfp.get_data().get_universe_size())))),
                                                            INT64_C(1)));
                                                        });
    return execute_for_lgrouping<indexing::BalancedFunctionalLengthGrouping, suffix_depth, sketch_vec, algorithm>(
      config,
      sfp,
      grouping);
  } else if (config.length_grouping == "wsqrt") {
    indexing::len_hist hist = indexing::build_length_histogram(sfp.get_data().get_records());
    indexing::BalancedFunctionalLengthGrouping grouping(hist,
                                                        [&]([[maybe_unused]] size_t _) {
                                                          // How much faster is scanning compared to switching between lists?
                                                          constexpr int64_t scan_bias = 2;
                                                          return static_cast<size_t>(std::max(
                                                            static_cast<int64_t>(std::round(std::sqrt(
                                                              1. * sfp.get_data().max_candidate_pairs() /
                                                              static_cast<double>(
                                                                scan_bias * sfp.get_data().token_count())))),
                                                            INT64_C(1)));
                                                        });
    return execute_for_lgrouping<indexing::BalancedFunctionalLengthGrouping, suffix_depth, sketch_vec, algorithm>(
      config,
      sfp,
      grouping);
  } else if (config.length_grouping == "fixed") {
    indexing::len_hist hist = indexing::build_length_histogram(sfp.get_data().get_records());
    indexing::FixedWidthLengthGrouping grouping(hist, 3);
    return execute_for_lgrouping<indexing::FixedWidthLengthGrouping, suffix_depth, sketch_vec, algorithm>(
      config,
      sfp,
      grouping);
  } else {
    indexing::len_hist hist = indexing::build_length_histogram(sfp.get_data().get_records());
    indexing::AIOLengthGrouping grouping(hist);
    return execute_for_lgrouping<indexing::AIOLengthGrouping, suffix_depth, sketch_vec, algorithm>(
      config,
      sfp,
      grouping);
  }
}

template <int32_t sketch_vec, Algorithm algorithm>
nlohmann::json execute_for_sketch_vec(Config& config, data::SetFileParser& sfp) {
  nlohmann::json result;

  switch (config.suffix_depth) {
  case 0:
    result = execute_for_suffix<0, sketch_vec, algorithm>(config, sfp);
    break;
  case 2:
    result = execute_for_suffix<2, sketch_vec, algorithm>(config, sfp);
    break;
  case 4:
    result = execute_for_suffix<4, sketch_vec, algorithm>(config, sfp);
    break;
  case 6:
    result = execute_for_suffix<6, sketch_vec, algorithm>(config, sfp);
    break;
  case 8:
    result = execute_for_suffix<8, sketch_vec, algorithm>(config, sfp);
    break;
  case 10:
    result = execute_for_suffix<10, sketch_vec, algorithm>(config, sfp);
    break;
  }

  return result;
}

template <Algorithm algorithm>
nlohmann::json execute_for_algorithm(Config& config, data::SetFileParser& sfp) {
  nlohmann::json result;

  switch (config.sketch_vec) {
  case 0:
    result = execute_for_sketch_vec<0, algorithm>(config, sfp);
    break;
  default:
    result = execute_for_sketch_vec<1, algorithm>(config, sfp);
    break;
  }

  return result;
}

nlohmann::json execute_transformation(Config& config, data::SetFileParser& sfp) {
  nlohmann::json result;

  timing::RealTimer outer_time;

  outer_time.start();
  knn::transformation::Transformation transformation(sfp.get_data(), config.k);
  if (config.mode == "sample") {
    transformation.sample_and_query(config.sample_size, config.seed, config.concurrency);
  } else {
    transformation.join(config.concurrency);
  }
  outer_time.stop();

  result["statistics"] = transformation.get_json_statistics();
  result["meta"] = get_metadata<TRANSFORMATION>(config, sfp);
  result["meta"]["outer_time"] = outer_time.get();

  return result;
}

template <class Baseline>
nlohmann::json execute_baseline(Config& config, data::SetFileParser& sfp) {
  nlohmann::json result;

  timing::RealTimer outer_time;

  outer_time.start();
  Baseline baseline(sfp.get_data(), config.k, config.min_batch_size, config.max_batch_size);
  if (config.mode == "sample") {
    baseline.sample_and_query(config.sample_size, config.seed, config.concurrency);
  } else {
    baseline.join(config.concurrency);
  }
  outer_time.stop();

  result["statistics"] = baseline.get_json_statistics();
  result["meta"] = get_metadata<BASELINE>(config, sfp);
  result["meta"]["outer_time"] = outer_time.get();

  return result;
}

template <bool deletion>
nlohmann::json execute_partition_deletion(Config& config, data::SetFileParser& sfp) {
  nlohmann::json result;
  timing::RealTimer outer_time;

  knn::partition::PartJoin<similarity::JaccardSimilarity, deletion> pj(sfp.get_data(), config.k);
  if (config.mode == "sample") {
    pj.sample_and_query(config.sample_size, config.seed, config.concurrency);
  } else {
    pj.join(config.concurrency);
  }

  outer_time.stop();

  result["statistics"] = pj.get_json_statistics();
  result["meta"] = get_metadata<PARTITION>(config, sfp);
  result["meta"]["outer_time"] = outer_time.get();

  return result;
}

nlohmann::json execute_partition(Config& config, data::SetFileParser& sfp) {
  if (config.deletion || config.algorithm == "palloc") {
    return execute_partition_deletion<true>(config, sfp);
  } else {
    return execute_partition_deletion<false>(config, sfp);
  }
}

int32_t main(int32_t argc, char** argv) {
  Config config;

  if (!process_program_options(argc, argv, config)) {
    exit(-1);
  }

  data::SetFileParser sfp(config.input_file);
  sfp.parse();

  nlohmann::json result;

  if (config.algorithm == "full") {
    result = execute_for_algorithm<FULL>(config, sfp);
  } else if (config.algorithm == "slim") {
    result = execute_for_algorithm<SLIM>(config, sfp);
  } else if (config.algorithm == "topkbaseline") {
    result = execute_for_algorithm<TOPKBASELINE>(config, sfp);
  } else if (config.algorithm == "transformation") {
    result = execute_transformation(config, sfp);
  } else if (config.algorithm == "partition" || config.algorithm == "palloc") {
    result = execute_partition(config, sfp);
  } else if (config.algorithm == "baselinepp") {
    result = execute_baseline<knn::BaselinePPJoin<similarity::JaccardSimilarity>>(config, sfp);
  } else {
    // config.algorithm == "baseline"
    result = execute_baseline<knn::BaselineJoin<similarity::JaccardSimilarity>>(config, sfp);
  }

  std::cout << result.dump(4) << std::endl;
}