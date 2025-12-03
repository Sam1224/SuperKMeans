#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"

class UnrotateTest : public ::testing::Test {
  protected:
    void GenerateRandomVectors(
        size_t n,
        size_t d,
        std::vector<float>& output,
        unsigned int seed = 42
    ) {
        output.resize(n * d);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& val : output) {
            val = dist(rng);
        }
    }
};

TEST_F(UnrotateTest, RotateUnrotateInverse_LowDim) {
    // Test with low dimensions (uses orthonormal matrix rotation)
    const size_t d = 128;
    const size_t n = 100;

    std::vector<float> original;
    GenerateRandomVectors(n, d, original);

    // Create pruner (which has Rotate/Unrotate)
    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

    // Rotate
    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    // Unrotate
    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    // Compare original and recovered
    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

TEST_F(UnrotateTest, RotateUnrotateInverse_HighDim_DCT) {
    // Test with high dimensions (uses DCT rotation)
    const size_t d = 1024;
    const size_t n = 100;

    ASSERT_GE(d, skmeans::D_THRESHOLD_FOR_DCT_ROTATION)
        << "Test expects d >= D_THRESHOLD_FOR_DCT_ROTATION to use DCT rotation";

    std::vector<float> original;
    GenerateRandomVectors(n, d, original);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

TEST_F(UnrotateTest, RotateUnrotateInverse_MultipleDimensions) {
    // Test both rotation methods across different dimensions
    std::vector<size_t> dimensions = {50, 128, 256, 512, 768, 1024};
    const size_t n = 100;

    for (size_t d : dimensions) {
        std::vector<float> original;
        GenerateRandomVectors(n, d, original);

        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        std::vector<float> recovered(n * d);
        pruner.Unrotate(rotated.data(), recovered.data(), n);

        double max_error = 0.0;
        double sum_error = 0.0;
        for (size_t i = 0; i < n * d; ++i) {
            double error = std::abs(original[i] - recovered[i]);
            max_error = std::max(max_error, error);
            sum_error += error;
        }
        double avg_error = sum_error / (n * d);

        EXPECT_LT(max_error, 1e-4)
            << "Max error too large for d=" << d << " (using "
            << (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION ? "DCT" : "orthonormal matrix")
            << " rotation)";
        EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
    }
}

TEST_F(UnrotateTest, RotatePreservesNorm) {
    // Rotation should preserve vector norms (orthogonal transformation)
    const size_t d = 256;
    const size_t n = 50;

    std::vector<float> original;
    GenerateRandomVectors(n, d, original);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    // Check that norms are preserved
    for (size_t i = 0; i < n; ++i) {
        float original_norm = 0.0f;
        float rotated_norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            original_norm += original[i * d + j] * original[i * d + j];
            rotated_norm += rotated[i * d + j] * rotated[i * d + j];
        }
        original_norm = std::sqrt(original_norm);
        rotated_norm = std::sqrt(rotated_norm);

        EXPECT_NEAR(original_norm, rotated_norm, 1e-4)
            << "Rotation should preserve norm for vector " << i;
    }
}
