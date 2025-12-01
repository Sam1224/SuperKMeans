#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>

#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"
#include "bench_utils.h"

int main(int argc, char* argv[]) {
    // Choose dataset by name
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("sift");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist, glove200, glove100, glove50\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t n_queries = 1000;
    const size_t d = it->second.second;
    const size_t n_clusters = std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = 25;
    float sampling_fraction = 1.0;

    std::string path_root = std::string(CMAKE_SOURCE_DIR) + "/benchmarks";
    std::string filename = path_root + "/data_" + dataset + ".bin";
    std::string filename_queries = path_root + "/data_" + dataset + "_test.bin";

    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);

    std::cout << "=== SuperKMeans vs FAISS Real Dataset Benchmark ===" << std::endl << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "FAISS compile options: " << faiss::get_compile_options() << std::endl;
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl << std::endl;

    // Load data
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;

    try {
        data.resize(n * d);
        queries.resize(n_queries * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what() << "\n";
        return 1;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        std::cerr << "Please run bench_gen.py to generate the dataset files." << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();
    std::cout << "Loaded training data from " << filename << std::endl;

    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Warning: Failed to open " << filename_queries << ", skipping query benchmarks" << std::endl;
    } else {
        file_queries.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        file_queries.close();
        std::cout << "Loaded query data from " << filename_queries << std::endl;
    }

    std::cout << std::endl;

    // Benchmark 1: Training with queries (recall computation) - SuperKMeans only
    if (file_queries) {
        std::cout << "--- Benchmark: Training with Recall Computation (SuperKMeans only) ---" << std::endl;
        skmeans::SuperKMeansConfig config;
        config.iters = n_iters;
        config.sampling_fraction = sampling_fraction;
        config.verbose = false;
        config.n_threads = THREADS;
        config.objective_k = 100;
        config.ann_explore_fraction = 0.01f;
        config.unrotate_centroids = false;
        config.perform_assignments = false;
        config.early_termination = false;

        ankerl::nanobench::Bench()
            .epochs(1)
            .epochIterations(1)
            .run(dataset + "_SuperKMeans_WithQueries", [&]() {
                auto kmeans_state = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
                auto centroids = kmeans_state.Train(data.data(), n, queries.data(), n_queries);
                ankerl::nanobench::doNotOptimizeAway(centroids);
            });

        std::cout << std::endl;
    }

    // Benchmark 2: Training comparison - SuperKMeans vs FAISS
    {
        std::cout << "--- Benchmark: Training Comparison ---" << std::endl;

        // SuperKMeans
        std::cout << "SuperKMeans:" << std::endl;
        skmeans::SuperKMeansConfig config;
        config.iters = n_iters;
        config.sampling_fraction = sampling_fraction;
        config.verbose = false;
        config.n_threads = THREADS;
        config.unrotate_centroids = false;
        config.perform_assignments = false;
        config.early_termination = false;

        ankerl::nanobench::Bench()
            .epochs(1)
            .epochIterations(1)
            .run(dataset + "_SuperKMeans", [&]() {
                auto kmeans_state = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
                auto centroids = kmeans_state.Train(data.data(), n);
                ankerl::nanobench::doNotOptimizeAway(centroids);
            });

        std::cout << std::endl;

        // FAISS
        std::cout << "FAISS:" << std::endl;
        faiss::ClusteringParameters cp;
        cp.niter = n_iters;
        cp.verbose = false;
        cp.nredo = 1;

        ankerl::nanobench::Bench()
            .epochs(1)
            .epochIterations(1)
            .run(dataset + "_FAISS", [&]() {
                faiss::Clustering clus(d, n_clusters, cp);
                faiss::IndexFlatL2 index(d);
                clus.train(n, data.data(), index);
                ankerl::nanobench::doNotOptimizeAway(clus.centroids);
            });

        std::cout << std::endl;
    }

    // Benchmark 3: Different sampling fractions
    {
        std::cout << "--- Benchmark: Sampling Fraction Impact ---" << std::endl;
        std::vector<float> sampling_fractions = {0.1f, 0.25f, 0.5f, 1.0f};

        for (float sf : sampling_fractions) {
            skmeans::SuperKMeansConfig config;
            config.iters = n_iters;
            config.sampling_fraction = sf;
            config.verbose = false;
            config.n_threads = THREADS;
            config.unrotate_centroids = false;
            config.perform_assignments = false;

            std::string bench_name = dataset + "_Sampling_" + std::to_string(int(sf * 100)) + "pct";

            ankerl::nanobench::Bench()
                .epochs(1)
                .epochIterations(1)
                .run(bench_name, [&]() {
                    auto kmeans_state = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                        n_clusters, d, config
                    );
                    auto centroids = kmeans_state.Train(data.data(), n);
                    ankerl::nanobench::doNotOptimizeAway(centroids);
                });
        }

        std::cout << std::endl;
    }

    std::cout << "=== Benchmark Complete ===" << std::endl;
    return 0;
}
