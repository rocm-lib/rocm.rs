#include <hip/hip_runtime.h>
#include <stdint.h>

namespace detail {

constexpr uint32_t THREADS_X = 16;
constexpr uint32_t THREADS_Y = 16;
constexpr uint32_t TILE_M = 32;
constexpr uint32_t TILE_N = 32;
constexpr uint32_t TILE_K = 8;

template <typename T, bool TRANSPOSE_A>
__device__ __forceinline__ T load_a(
    const T* __restrict__ a,
    uint32_t row,
    uint32_t col,
    uint32_t lda
) {
    if constexpr (TRANSPOSE_A) {
        return a[col * lda + row];
    } else {
        return a[row * lda + col];
    }
}

template <typename T, bool TRANSPOSE_B>
__device__ __forceinline__ T load_b(
    const T* __restrict__ b,
    uint32_t row,
    uint32_t col,
    uint32_t ldb
) {
    if constexpr (TRANSPOSE_B) {
        return b[col * ldb + row];
    } else {
        return b[row * ldb + col];
    }
}

}  // namespace detail

#define DEFINE_GEMM_WRAPPER(NAME, TYPE, TRANSPOSE_A, TRANSPOSE_B) \
    extern "C" __global__ __launch_bounds__(detail::THREADS_X * detail::THREADS_Y) void NAME( \
        const TYPE* __restrict__ a, \
        const TYPE* __restrict__ b, \
        TYPE* __restrict__ c, \
        uint32_t m, \
        uint32_t n, \
        uint32_t k, \
        uint32_t lda, \
        uint32_t ldb, \
        uint32_t ldc, \
        uint32_t transpose_a, \
        uint32_t transpose_b \
    ) { \
        (void) transpose_a; \
        (void) transpose_b; \
        __shared__ TYPE a_tile[detail::TILE_M][detail::TILE_K + 1]; \
        __shared__ TYPE b_tile[detail::TILE_K][detail::TILE_N + 1]; \
        const uint32_t tx = threadIdx.x; \
        const uint32_t ty = threadIdx.y; \
        const uint32_t linear_tid = ty * detail::THREADS_X + tx; \
        const uint32_t block_row = blockIdx.y * detail::TILE_M; \
        const uint32_t block_col = blockIdx.x * detail::TILE_N; \
        const uint32_t row0 = block_row + ty; \
        const uint32_t row1 = block_row + ty + detail::THREADS_Y; \
        const uint32_t col0 = block_col + tx; \
        const uint32_t col1 = block_col + tx + detail::THREADS_X; \
        TYPE acc00 = TYPE(0); \
        TYPE acc01 = TYPE(0); \
        TYPE acc10 = TYPE(0); \
        TYPE acc11 = TYPE(0); \
        for (uint32_t k_base = 0; k_base < k; k_base += detail::TILE_K) { \
            const uint32_t a_tile_row = linear_tid / detail::TILE_K; \
            const uint32_t a_tile_col = linear_tid % detail::TILE_K; \
            const uint32_t global_a_row = block_row + a_tile_row; \
            const uint32_t global_a_col = k_base + a_tile_col; \
            a_tile[a_tile_row][a_tile_col] = \
                (global_a_row < m && global_a_col < k) \
                    ? detail::load_a<TYPE, TRANSPOSE_A>(a, global_a_row, global_a_col, lda) \
                    : TYPE(0); \
            const uint32_t b_tile_row = linear_tid / detail::TILE_N; \
            const uint32_t b_tile_col = linear_tid % detail::TILE_N; \
            const uint32_t global_b_row = k_base + b_tile_row; \
            const uint32_t global_b_col = block_col + b_tile_col; \
            b_tile[b_tile_row][b_tile_col] = \
                (global_b_row < k && global_b_col < n) \
                    ? detail::load_b<TYPE, TRANSPOSE_B>(b, global_b_row, global_b_col, ldb) \
                    : TYPE(0); \
            __syncthreads(); \
            _Pragma("unroll") \
            for (uint32_t kk = 0; kk < detail::TILE_K; ++kk) { \
                const TYPE a0 = a_tile[ty][kk]; \
                const TYPE a1 = a_tile[ty + detail::THREADS_Y][kk]; \
                const TYPE b0 = b_tile[kk][tx]; \
                const TYPE b1 = b_tile[kk][tx + detail::THREADS_X]; \
                acc00 += a0 * b0; \
                acc01 += a0 * b1; \
                acc10 += a1 * b0; \
                acc11 += a1 * b1; \
            } \
            __syncthreads(); \
        } \
        if (row0 < m && col0 < n) { \
            c[row0 * ldc + col0] = acc00; \
        } \
        if (row0 < m && col1 < n) { \
            c[row0 * ldc + col1] = acc01; \
        } \
        if (row1 < m && col0 < n) { \
            c[row1 * ldc + col0] = acc10; \
        } \
        if (row1 < m && col1 < n) { \
            c[row1 * ldc + col1] = acc11; \
        } \
    }

DEFINE_GEMM_WRAPPER(sgemm_nn, float, false, false)
DEFINE_GEMM_WRAPPER(sgemm_nt, float, false, true)
DEFINE_GEMM_WRAPPER(sgemm_tn, float, true, false)
DEFINE_GEMM_WRAPPER(sgemm_tt, float, true, true)

DEFINE_GEMM_WRAPPER(dgemm_nn, double, false, false)
DEFINE_GEMM_WRAPPER(dgemm_nt, double, false, true)
DEFINE_GEMM_WRAPPER(dgemm_tn, double, true, false)
DEFINE_GEMM_WRAPPER(dgemm_tt, double, true, true)

#undef DEFINE_GEMM_WRAPPER
