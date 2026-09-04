#include "vector_add.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <vector>

using trail::reference::vector_add;

TEST_CASE("vector_add matches elementwise sum for typical values", "[reference][vector_add]") {
    const std::vector<float> lhs{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> rhs{10.0F, 20.0F, 30.0F, 40.0F};
    REQUIRE(vector_add(lhs, rhs) == std::vector<float>{11.0F, 22.0F, 33.0F, 44.0F});
}

TEST_CASE("vector_add handles the empty vector", "[reference][vector_add]") {
    REQUIRE(vector_add({}, {}).empty());
}

TEST_CASE("vector_add handles zeros", "[reference][vector_add]") {
    const std::vector<float> lhs{0.0F, 0.0F};
    const std::vector<float> rhs{0.0F, -0.0F};
    REQUIRE(vector_add(lhs, rhs) == std::vector<float>{0.0F, 0.0F});
}

TEST_CASE("vector_add handles negative values", "[reference][vector_add]") {
    const std::vector<float> lhs{-5.0F, 5.0F};
    const std::vector<float> rhs{5.0F, -5.0F};
    REQUIRE(vector_add(lhs, rhs) == std::vector<float>{0.0F, 0.0F});
}

TEST_CASE("vector_add handles a non-power-of-two length", "[reference][vector_add]") {
    const std::size_t length = 37;
    const std::vector<float> lhs(length, 1.0F);
    const std::vector<float> rhs(length, 2.0F);
    const auto result = vector_add(lhs, rhs);
    REQUIRE(result.size() == length);
    for (float value : result) {
        REQUIRE(value == 3.0F);
    }
}

TEST_CASE("vector_add handles very large magnitude values", "[reference][vector_add]") {
    const std::vector<float> lhs{std::numeric_limits<float>::max() / 2};
    const std::vector<float> rhs{std::numeric_limits<float>::max() / 2};
    REQUIRE(vector_add(lhs, rhs)[0] == std::numeric_limits<float>::max());
}
