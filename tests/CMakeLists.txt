# Test suite for SuperKMeans

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Common test dependencies
set(TEST_COMMON_LIBS
    gtest
    gtest_main
    "-framework Accelerate"
)

if (FFTW_FOUND)
    list(APPEND TEST_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
endif()

# Test: Unrotate functionality
add_executable(test_unrotate test_unrotate.cpp)
target_link_libraries(test_unrotate PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME UnrotateTest COMMAND test_unrotate)

# Test: Assign functionality
add_executable(test_assign test_assign.cpp)
target_link_libraries(test_assign PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME AssignTest COMMAND test_assign)

# Test: SuperKMeans core functionality
add_executable(test_superkmeans test_superkmeans.cpp)
target_link_libraries(test_superkmeans PRIVATE ${TEST_COMMON_LIBS})
add_test(NAME SuperKMeansTest COMMAND test_superkmeans)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(test_unrotate)
gtest_discover_tests(test_assign)
gtest_discover_tests(test_superkmeans)
