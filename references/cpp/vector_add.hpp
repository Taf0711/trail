#pragma once

#include <cstddef>
#include <vector>

namespace trail::reference {

// Boring, correctness-first CPU oracle for elementwise vector addition.
// Prioritizes clarity over speed; this is what the CUDA kernel is checked against.
inline std::vector<float> vector_add(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    std::vector<float> result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

}  // namespace trail::reference
