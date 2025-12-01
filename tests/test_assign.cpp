#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

namespace {

// Helper function to generate synthetic clusterable data
std::vector<float> make_blobs(size_t n_samples, size_t n_features, size_t n_centers, unsigned int random_state = 42) {
    std::mt19937 gen(random_state);

    // Random cluster centers
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(n_features));
    for (auto& c : centers)
        for (auto& x : c)
            x = center_dist(gen);

    // Distributions for choosing cluster and spreading points
    std::uniform_int_distribution<size_t> cluster_dist(0, n_centers - 1);
    std::normal_distribution<float> point_dist(0.0f, 1.0f);

    // Flattened result: row-major layout [sample0_dim0, sample0_dim1, ..., sampleN_dimD]
    std::vector<float> data;
    data.reserve(n_samples * n_features);

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& center = centers[cluster_dist(gen)];
        for (size_t j = 0; j < n_features; ++j)
            data.push_back(center[j] + point_dist(gen));
    }

    return data;
}

} // anonymous namespace

class AssignTest : public ::testing::Test {
protected:
    void SetUp() override {
        omp_set_num_threads(10);
    }
};

TEST_F(AssignTest, AssignMatchesTrainAssignments_SIFT_NoSampling) {
    const size_t n = 1000000;
    const size_t d = 128;
    const size_t n_clusters = std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    const int n_iters = 5;
    const float sampling_fraction = 1.0f;  // No sampling - use all data

    std::string path_root = std::string(CMAKE_SOURCE_DIR);
    std::string filename = path_root + "/data_sift.bin";

    // Load data
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    data.resize(n * d);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        GTEST_SKIP() << "Data file not found: " << filename;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    // Create and train k-means
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = sampling_fraction;
    config.verbose = false;
    config.n_threads = 10;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(n_clusters, d, config);

    auto centroids = kmeans.Train(data.data(), n);

    // Get the internal assignments from Train()
    const auto& train_assignments = kmeans._assignments;
    ASSERT_EQ(train_assignments.size(), n);

    // Now run Assign() with raw data and unrotated centroids
    auto assign_assignments = kmeans.Assign(
        data.data(),
        centroids.data(),
        n, n_clusters
    );
    ASSERT_EQ(assign_assignments.size(), n);

    // Compare assignments
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (train_assignments[i] != assign_assignments[i]) {
            ++mismatches;
        }
    }

    double mismatch_pct = 100.0 * static_cast<double>(mismatches) / static_cast<double>(n);

    // Check if within tolerance (0.01%)
    const double tolerance = 0.01;
    EXPECT_LE(mismatch_pct, tolerance)
        << "Mismatch rate (" << mismatch_pct << "%) exceeds tolerance (" << tolerance << "%)";
}

TEST_F(AssignTest, AssignMatchesTrainAssignments_SyntheticClusters) {
    // Test with synthetic clusterable data
    const size_t n = 100000;
    const size_t d = 128;
    const size_t n_clusters = 1024;
    const int n_iters = 25;
    const float sampling_fraction = 1.0f;

    // Generate synthetic clusterable data
    std::vector<float> data = make_blobs(n, d, n_clusters, 42);

    // Create and train k-means
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = sampling_fraction;
    config.verbose = false;
    config.n_threads = 10;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(n_clusters, d, config);

    auto centroids = kmeans.Train(data.data(), n);

    const auto& train_assignments = kmeans._assignments;
    auto assign_assignments = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::Assign(
        data.data(), centroids.data(), n, n_clusters, d
    );

    // For synthetic clusterable data with no sampling, assignments should match exactly or very closely
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (train_assignments[i] != assign_assignments[i]) {
            ++mismatches;
        }
    }

    double mismatch_pct = 100.0 * static_cast<double>(mismatches) / static_cast<double>(n);
    EXPECT_LE(mismatch_pct, 0.01)
        << "Synthetic clusterable data should have very low mismatch rate, got " << mismatch_pct << "%";
}
