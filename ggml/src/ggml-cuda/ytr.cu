#include "ytr.cuh"
#include "unary.cuh"
#include "top-k.cuh"
#include "ggml-impl.h"

#include <cmath>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Tensor data of a YTR weight (little endian, offsets from the start of the data):
//   64-byte header (ytr_header), then the parts it points to:
//   lut   4096 x half2   code table: state -> two weights
//   scale rows x half    one scale per row
//   sin   cols bits      input signs (bit set = -1)
//   sout  rows bits      output signs, only if flags & 1
//   rmap  rows x int32   row r of the result is row rmap[r] of the coded matrix, only if flags & 2
//   cmap  cols x int32   column c of the coded matrix reads input element cmap[c], only if flags & 4
//   codes rows x (cols/256) x (4*bits) uint32; a sequence of 256 columns is 128 steps of `bits` bits, LSB first. Row major, or with
//         flags & 8 in tiles of 16 rows: [row/16][sequence][16-byte chunk][row%16][4 words], so a warp's loads are contiguous;
//         flags & 16 in the lane layout: per 16-row tile and sequence, lane l = 4 gi + q holds words K q .. K q + K - 1 of rows gi and
//         gi + 8 (2K words = K/2 uint4), uint4 u of lane l at ((tile * seqs + s) * K/2 + u) * 32 + l
// Weights: W = S_out H diag(scale) Q H S_in, with H the orthonormal 512-point Walsh-Hadamard transform per block of 512.
// Q[r][256 s + 2 j + t] = lut[state_j].t, state_j = bits [bits*(j+1) - 12, bits*(j+1)) of sequence s (zeros below bit 0).
//
// Up to 64 input rows (lane layout), default int8 path as in YASAKANI: int8 code table (one scale per tensor), int8 inputs with one
// scale per 512 values (rotated by the producing kernel), mma m16n8k32 s8. Each warp decodes 16 rows; lane q of a quad owns the
// contiguous quarter q of each 256-column sequence, so the input stays in natural column order. Split-K workers add partial sums
// to a float buffer; a finishing kernel applies the output rotation, the signs and the row map. ggml_cuda_ytr_try_fuse runs
// products that share an input in one launch and folds the following ADD + RMS_NORM + MUL (or SWIGLU) and the next product's
// input rotation into that finishing kernel. Kernels use PDL: weights, signs and the code table are read before the wait.
// More than 64 input rows: weights decoded to int8 once per call, then an int8 GEMM (per-512 input scales as above).
// GGML_YTR_F16=1: the earlier FP16 kernel (table and inputs in FP16). GGML_YTR_REF=1: decode to F16 + cuBLAS.
// GGML_YTR_FUSE=0: one product per launch, plain finish. GGML_YTR_WANT: split-K target, warps per SM (32).

struct ytr_header {
    uint32_t magic;
    uint16_t bits;
    uint16_t flags;
    uint32_t rows, cols;
    uint32_t lut_off, scale_off, sin_off, sout_off, rmap_off, cmap_off, code_off;
    uint32_t reserved[5];
};
static_assert(sizeof(ytr_header) == 64, "ytr_header size");

static constexpr uint32_t YTR_MAGIC   = 0x31525459u; // "YTR1"
static constexpr int      YTR_NMAX    = 64;          // most input rows of the fast path
static constexpr int      YTR_RMAX    = 32768;       // most rows of a weight with an output rotation or a row map
static constexpr int      YTR_CMAX    = 32768;       // most columns
static constexpr int64_t  YTR_ACC     = (int64_t) YTR_NMAX * 65536; // floats of the partial-sum buffer (rows x input rows of a launch)
static constexpr int      YTR_MAX_SEG = 4;           // most products in one launch
static constexpr float    YTR_HS      = 0.044194173824159216f;      // 1 / sqrt(512)
static constexpr int      YTR_XROW    = 528;         // bytes per staged input row of a group (512 + 16: reads of 8 rows hit all banks)

struct ytr_info {
    ytr_header h;
    uint64_t   in_key = 0;    // equal keys = same input transform (signs, column map)
    int *      rinv   = nullptr; // inverse of rmap on the device: coded row -> result row
    uint16_t * lut8   = nullptr; // int8 code table: two int8 weights per state, scale gamma
    float      gamma  = 1.0f;
};

static std::mutex                                g_ytr_mutex;
static std::unordered_map<const void *, ytr_info> g_ytr_info;
static std::unordered_set<const ggml_tensor *>   g_ytr_done;   // products of this graph run already computed by a combined launch
static uint64_t                                  g_ytr_epoch = 0;
struct ytr_job;
static void ytr_side_clear();
static void ytr_shead_clear();
static std::unordered_map<const ggml_tensor *, int> g_ytr_defer_n;   // see g_ytr_defer

void ggml_cuda_ytr_prepare(const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(g_ytr_mutex);
    g_ytr_epoch++;
    g_ytr_done.clear();
    ytr_side_clear();
    ytr_shead_clear();
    GGML_ASSERT(g_ytr_defer_n.empty() && "deferred YTR product not finished in the previous graph");
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if ((node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_GET_ROWS) || !ggml_is_ytr(node->src[0]->type)) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (g_ytr_info.count(src0->data)) {
            continue;
        }
        ytr_info inf;
        ytr_header & h = inf.h;
        const size_t nbytes = ggml_nbytes(src0);
        GGML_ASSERT(nbytes >= sizeof(h));
        CUDA_CHECK(cudaMemcpy(&h, src0->data, sizeof(h), cudaMemcpyDeviceToHost));
        GGML_ASSERT(h.magic == YTR_MAGIC && "YTR tensor without header");
        GGML_ASSERT((int64_t) h.rows == src0->ne[1] && (int64_t) h.cols == src0->ne[0]);
        GGML_ASSERT((h.bits == 6 && src0->type == GGML_TYPE_YTR3) || (h.bits == 8 && src0->type == GGML_TYPE_YTR4));
        GGML_ASSERT((h.flags & ~31u) == 0 && (h.flags & 24) != 24);
        GGML_ASSERT(h.cols > 0 && h.cols % 512 == 0 && h.cols <= YTR_CMAX && h.rows > 0 && h.rows % 128 == 0 && (!(h.flags & 3) || (h.rows % 512 == 0 && h.rows <= YTR_RMAX)));
        const auto check_range = [&](uint32_t off, uint64_t bytes, size_t align) {
            GGML_ASSERT(off >= sizeof(h) && off % align == 0 && off <= nbytes && bytes <= nbytes - off);
        };
        check_range(h.lut_off, 16384, sizeof(half2));
        check_range(h.scale_off, (uint64_t) h.rows * sizeof(half), sizeof(half));
        check_range(h.sin_off, h.cols / 8, sizeof(uint4));
        check_range(h.code_off, (uint64_t) h.rows * (h.cols / 256) * 16 * h.bits, sizeof(uint4));
        if (h.flags & 1) { check_range(h.sout_off, h.rows / 8, sizeof(uint4)); }
        if (h.flags & 2) { check_range(h.rmap_off, (uint64_t) h.rows * sizeof(int32_t), sizeof(int32_t)); }
        if (h.flags & 4) { check_range(h.cmap_off, (uint64_t) h.cols * sizeof(int32_t), sizeof(int32_t)); }
        std::vector<uint8_t> sin(h.cols / 8);
        CUDA_CHECK(cudaMemcpy(sin.data(), (const uint8_t *) src0->data + h.sin_off, sin.size(), cudaMemcpyDeviceToHost));
        uint64_t key = 1469598103934665603ull ^ h.cols;
        for (uint8_t b : sin) {
            key = (key ^ b) * 1099511628211ull;
        }
        if (h.flags & 4) {
            std::vector<int32_t> cmap(h.cols);
            CUDA_CHECK(cudaMemcpy(cmap.data(), (const uint8_t *) src0->data + h.cmap_off, cmap.size() * 4, cudaMemcpyDeviceToHost));
            for (int32_t c : cmap) {
                GGML_ASSERT(c >= 0 && (uint32_t) c < h.cols);
                key = (key ^ (uint32_t) c) * 1099511628211ull;
            }
        }
        inf.in_key = key;
        if (h.flags & 2) {
            std::vector<int32_t> rmap(h.rows), rinv(h.rows, -1);
            CUDA_CHECK(cudaMemcpy(rmap.data(), (const uint8_t *) src0->data + h.rmap_off, rmap.size() * 4, cudaMemcpyDeviceToHost));
            for (uint32_t r = 0; r < h.rows; ++r) {
                GGML_ASSERT(rmap[r] >= 0 && (uint32_t) rmap[r] < h.rows && rinv[rmap[r]] == -1);
                rinv[rmap[r]] = (int32_t) r;
            }
            CUDA_CHECK(cudaMalloc(&inf.rinv, h.rows * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(inf.rinv, rinv.data(), h.rows * sizeof(int), cudaMemcpyHostToDevice));
        }
        {   // int8 table as YASAKANI builds it: one scale per tensor, gamma = max |value| / 127
            std::vector<uint16_t> l16(8192), l8(4096);
            CUDA_CHECK(cudaMemcpy(l16.data(), (const uint8_t *) src0->data + h.lut_off, 16384, cudaMemcpyDeviceToHost));
            float mx = 0.0f;
            for (uint16_t u : l16) { mx = std::max(mx, std::fabs(__half2float(*(const __half *) &u))); }
            inf.gamma = mx > 0.0f ? mx / 127.0f : 1.0f;
            for (int i = 0; i < 4096; ++i) {
                int q0 = (int) std::nearbyint(__half2float(*(const __half *) &l16[2 * i]) / inf.gamma);
                int q1 = (int) std::nearbyint(__half2float(*(const __half *) &l16[2 * i + 1]) / inf.gamma);
                q0 = std::max(-127, std::min(127, q0)); q1 = std::max(-127, std::min(127, q1));
                l8[i] = (uint16_t) ((uint8_t) (int8_t) q0 | ((uint16_t) (uint8_t) (int8_t) q1 << 8));
            }
            CUDA_CHECK(cudaMalloc(&inf.lut8, 8192));
            CUDA_CHECK(cudaMemcpy(inf.lut8, l8.data(), 8192, cudaMemcpyHostToDevice));
        }
        g_ytr_info[src0->data] = inf;
    }
}

static ytr_info ytr_get_info(const ggml_tensor * src0) {
    std::lock_guard<std::mutex> lock(g_ytr_mutex);
    auto it = g_ytr_info.find(src0->data);
    GGML_ASSERT(it != g_ytr_info.end() && "YTR header not read: ggml_cuda_ytr_prepare was not called");
    return it->second;
}

static __device__ __forceinline__ float ytr_sign(const uint8_t * bits, int i) {
    return (bits[i >> 3] >> (i & 7)) & 1 ? -1.0f : 1.0f;
}

// in-place orthonormal-up-to-scale Walsh-Hadamard transform of 512 values in shared memory, any blockDim.x dividing 256
static __device__ __forceinline__ void ytr_fwht512(float * v) {
#pragma unroll
    for (int h = 1; h < 512; h <<= 1) {
        for (int t = threadIdx.x; t < 256; t += blockDim.x) {
            const int lo = ((t & ~(h - 1)) << 1) | (t & (h - 1));
            const float a = v[lo], b = v[lo + h];
            v[lo] = a + b; v[lo + h] = a - b;
        }
        __syncthreads();
    }
}

// xt[n][c] = (H S_in x_code)[c] as half, x_code[c] = x[n][cmap[c]]; grid (cols/512, N).
// np > 0: fragment layout of the fast kernel instead, per 256-column sequence s: [s][j 0..7][n 0..np-1][q 0..3][8 halves] holding
// columns 64 q + 8 j .. 64 q + 8 j + 7 (lane q's quarter): one 16-byte load gives a lane the inputs of two MMA steps.
static __global__ void ytr_in_kernel(const float * __restrict__ x, const int64_t sx, const uint8_t * __restrict__ base, const ytr_header h,
                                     half * __restrict__ xt, const int np) {
    __shared__ float v[512];
    const int64_t n  = blockIdx.y;
    const int     c0 = blockIdx.x * 512;
    const uint8_t * sin  = base + h.sin_off;
    const int32_t * cmap = (h.flags & 4) ? (const int32_t *) (base + h.cmap_off) : nullptr;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        const int c = c0 + i;
        v[i] = x[n * sx + (cmap ? cmap[c] : c)] * ytr_sign(sin, c);
    }
    __syncthreads();
    ytr_fwht512(v);
    if (np == 0) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) {
            xt[n * h.cols + c0 + i] = __float2half(v[i] * 0.044194173824159216f);
        }
        return;
    }
    half2 * x2 = (half2 *) xt;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {      // half2 pairs of this 512-column block
        const int c  = c0 + 2 * i;
        const int s  = c / 256, h2 = (c % 256) / 2;
        const int k  = 2 * h2, q = k / 64, j = (k % 64) / 8, e = (k % 8) / 2;
        x2[((((int64_t) s * 8 + j) * np + n) * 4 + q) * 4 + e] = __floats2half2_rn(v[2 * i] * 0.044194173824159216f, v[2 * i + 1] * 0.044194173824159216f);
    }
}

// ---- reference path ----------------------------------------------------------------------------------------------------------

// Q rows [r0, r0 + nr) as half: one thread per step (two weights)
static __global__ void ytr_decode_kernel(const uint8_t * __restrict__ base, const ytr_header h, const int64_t r0, const int64_t nr,
                                         half2 * __restrict__ q) {
    const int64_t seqs = h.cols / 256;
    const int64_t idx  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr * seqs * 128) {
        return;
    }
    const int     j = idx % 128;
    const int64_t s = (idx / 128) % seqs;
    const int64_t r = r0 + idx / (128 * seqs);
    const int     K = h.bits;
    const uint32_t * cw = (const uint32_t *) (base + h.code_off);
    auto word_at = [&](const int w) -> uint32_t {
        if (h.flags & 8) {
            return cw[((((r / 16) * seqs + s) * K + w / 4) * 16 + r % 16) * 4 + w % 4];
        }
        if (h.flags & 16) {
            const int64_t ri = r % 16, lane = (ri % 8) * 4 + w / K, idx = (ri / 8) * K + w % K;
            return cw[((((r / 16) * seqs + s) * (K / 2) + idx / 4) * 32 + lane) * 4 + idx % 4];
        }
        return cw[(r * seqs + s) * 4 * K + w];
    };
    const int e = K * (j + 1);
    uint32_t st;
    if (e < 12) {
        st = (word_at(0) & ((1u << e) - 1u)) << (12 - e);
    } else {
        const int start = e - 12, word = start >> 5, sh = start & 31;
        uint32_t v = word_at(word) >> sh;
        if (sh > 20) {
            v |= word_at(word + 1) << (32 - sh);
        }
        st = v & 0xFFFu;
    }
    const float2 l  = __half22float2(((const half2 *) (base + h.lut_off))[st]);
    const float  sc = __half2float(((const half *) (base + h.scale_off))[r]);
    q[idx] = __floats2half2_rn(l.x * sc, l.y * sc);
}

// y[n][r] = (S_out H y_code)[r] in place, per block of 512 rows; grid (rows/512, N)
static __global__ void ytr_out_kernel(float * __restrict__ y, const int64_t sy, const uint8_t * __restrict__ base, const ytr_header h) {
    __shared__ float v[512];
    float * yb = y + blockIdx.y * sy + blockIdx.x * 512;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        v[i] = yb[i];
    }
    __syncthreads();
    ytr_fwht512(v);
    const uint8_t * sout = base + h.sout_off;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        yb[i] = v[i] * 0.044194173824159216f * ytr_sign(sout, blockIdx.x * 512 + i);
    }
}

// y[n][r] = t[n][rmap[r]]
static __global__ void ytr_gather_kernel(const float * __restrict__ t, float * __restrict__ y, const int64_t sy, const int32_t * __restrict__ rmap,
                                         const int64_t rows, const int64_t n_tok) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_tok) {
        return;
    }
    const int64_t n = idx / rows, r = idx % rows;
    y[n * sy + r] = t[n * rows + rmap[r]];
}

static void ytr_mul_mat_ref(ggml_backend_cuda_context & ctx, const ytr_header & h, const uint8_t * base, const half * xt, const int64_t n_tok,
                            float * dst, const int64_t sy) {
    const int64_t rows = h.rows, cols = h.cols;
    cudaStream_t stream = ctx.stream();
    const int64_t chunk = std::max<int64_t>(512, ((int64_t) 1 << 27) / cols / 512 * 512);
    ggml_cuda_pool_alloc<half> q(ctx.pool(), std::min(chunk, rows) * cols);
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    const float alpha = 1.0f, beta = 0.0f;
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, rows - r0);
        const int64_t nt = nr * (cols / 256) * 128;
        ytr_decode_kernel<<<(nt + 255) / 256, 256, 0, stream>>>(base, h, r0, nr, (half2 *) q.get());
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, nr, n_tok, cols,
                                  &alpha, q.get(), CUDA_R_16F, cols, xt, CUDA_R_16F, cols,
                                  &beta, dst + r0, CUDA_R_32F, sy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    if (h.flags & 1) {
        ytr_out_kernel<<<dim3(rows / 512, n_tok), 256, 0, stream>>>(dst, sy, base, h);
        CUDA_CHECK(cudaGetLastError());
    }
    if (h.flags & 2) {
        ggml_cuda_pool_alloc<float> t(ctx.pool(), n_tok * rows);
        CUDA_CHECK(cudaMemcpy2DAsync(t.get(), rows * sizeof(float), dst, sy * sizeof(float), rows * sizeof(float), n_tok,
                                     cudaMemcpyDeviceToDevice, stream));
        ytr_gather_kernel<<<(rows * n_tok + 255) / 256, 256, 0, stream>>>(t.get(), dst, sy, (const int32_t *) (base + h.rmap_off), rows, n_tok);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ---- fast path ---------------------------------------------------------------------------------------------------------------

static __device__ __forceinline__ void ytr_mma(float * c, const uint32_t * a, const uint32_t * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// byte offset (index * 4) into the code table of the 12-bit window at a compile-time bit position
template <int POS, int NW>
static __device__ __forceinline__ uint32_t ytr_win(const uint32_t * u) {
    constexpr int word = POS >> 5, sh = POS & 31;
    if constexpr (sh < 2) {
        return (u[word] << (2 - sh)) & 0x3FFCu;
    } else if constexpr (sh <= 20 || word + 1 >= NW) {
        return (u[word] >> (sh - 2)) & 0x3FFCu;
    } else {
        return __funnelshift_r(u[word], u[word + 1], sh - 2) & 0x3FFCu;
    }
}

static __device__ __forceinline__ uint32_t ytr_lut(const uint32_t * s_lut, const uint32_t off) {
    return *(const uint32_t *) ((const uint8_t *) s_lut + off);
}

// MMA t of a 256-column sequence in lane-quarter order: lane q owns steps 32 q .. 32 q + 31 (columns 64 q .. 64 q + 63) of rows g
// and g + 8 and gives MMA t its steps 2t and 2t+1 (the input uses the same column order). With the 12 - K boundary bits below
// bit 0, the window of local step i starts at bit K i of the extended stream.
template <int K, int T2>
static __device__ __forceinline__ void ytr_qstep(const uint32_t * ea, const uint32_t * eb, const uint32_t * s_lut, const uint32_t b0,
                                                 const uint32_t b1, float * c) {
    constexpr int NE = K + 1;
    uint32_t a[4];
    a[0] = ytr_lut(s_lut, ytr_win<K * (2 * T2),     NE>(ea));
    a[1] = ytr_lut(s_lut, ytr_win<K * (2 * T2),     NE>(eb));
    a[2] = ytr_lut(s_lut, ytr_win<K * (2 * T2 + 1), NE>(ea));
    a[3] = ytr_lut(s_lut, ytr_win<K * (2 * T2 + 1), NE>(eb));
    const uint32_t b[2] = {b0, b1};
    ytr_mma(c, a, b);
}

template <int K, int NT, int J>
struct ytr_steps {
    static __device__ __forceinline__ void run(const uint32_t * ea, const uint32_t * eb, const uint32_t * s_lut, const uint4 * xq,
                                               const int np, const int n_tok, const int g, float (*c)[4]) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n = nt * 8 + g;
            const uint4 bq = n < n_tok ? xq[((int64_t) J * np + n) * 4] : make_uint4(0u, 0u, 0u, 0u);
            ytr_qstep<K, 2 * J>    (ea, eb, s_lut, bq.x, bq.y, c[nt]);
            ytr_qstep<K, 2 * J + 1>(ea, eb, s_lut, bq.z, bq.w, c[nt]);
        }
        ytr_steps<K, NT, J + 1>::run(ea, eb, s_lut, xq, np, n_tok, g, c);
    }
};
template <int K, int NT>
struct ytr_steps<K, NT, 8> {
    static __device__ __forceinline__ void run(const uint32_t *, const uint32_t *, const uint32_t *, const uint4 *, const int, const int, const int, float (*)[4]) {}
};


// tile = 128 rows (8 warps x 16) x kc blocks of 512 columns; persistent blocks walk the tiles (row tile fastest).
// Code words go from global memory straight to registers; only the code table is in shared memory.
template <int K, int NT>
static __global__ void __launch_bounds__(256) ytr_mm_kernel(const uint8_t * __restrict__ base, const ytr_header h, const half * __restrict__ xr,
        const int n_tok, const int kc, const int n_tiles, float * __restrict__ acc, int * __restrict__ cnt, const int * __restrict__ rinv,
        float * __restrict__ dst, const int64_t sy, const bool ext_fin) {
    __shared__ uint32_t s_lut[4096];
    __shared__ int      s_last;

    const uint32_t * lut_g = (const uint32_t *) (base + h.lut_off);
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        s_lut[i] = lut_g[i];
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const int seqs = h.cols / 256, n_kb = h.cols / 512, n_rt = h.rows / 128, n_kc = (n_kb + kc - 1) / kc;
    const uint4 *    codes4 = (const uint4 *) (base + h.code_off);
    const half *     scale = (const half *) (base + h.scale_off);
    const bool       direct = !(h.flags & 3);
    const int        prev   = (lane & ~3) | ((q + 3) & 3);   // lane holding the previous quarter
    const int        np     = (n_tok + 7) / 8 * 8;

    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int rt = tile % n_rt, kci = tile / n_rt;
        const int ra = rt * 128 + warp * 16 + g, rb = ra + 8;
        float c[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
        }
        const int s1  = min((kci + 1) * kc, n_kb) * 2;
        const int t16 = ra / 16;
        for (int s = kci * kc * 2; s < s1; ++s) {
            constexpr int T = 12 - K;
            uint32_t w[2 * K];
            const uint4 * pl = codes4 + ((int64_t) t16 * seqs + s) * (K / 2) * 32 + lane;
#pragma unroll
            for (int u = 0; u < K / 2; ++u) {
                const uint4 v = __ldg(pl + u * 32);
                w[4 * u] = v.x; w[4 * u + 1] = v.y; w[4 * u + 2] = v.z; w[4 * u + 3] = v.w;
            }
            const uint32_t pa = __shfl_sync(0xFFFFFFFFu, w[K - 1] >> (32 - T), prev);
            const uint32_t pb = __shfl_sync(0xFFFFFFFFu, w[2 * K - 1] >> (32 - T), prev);
            uint32_t ea[K + 1], eb[K + 1];
            ea[0] = (q ? pa : 0u) | (w[0] << T);
            eb[0] = (q ? pb : 0u) | (w[K] << T);
#pragma unroll
            for (int i = 1; i < K; ++i) {
                ea[i] = __funnelshift_l(w[i - 1], w[i], T);
                eb[i] = __funnelshift_l(w[K + i - 1], w[K + i], T);
            }
            ea[K] = w[K - 1] >> (32 - T);
            eb[K] = w[2 * K - 1] >> (32 - T);
            ytr_steps<K, NT, 0>::run(ea, eb, s_lut, (const uint4 *) xr + (int64_t) s * 8 * np * 4 + q, np, n_tok, g, c);
        }

        const float sc0 = __half2float(scale[ra]), sc1 = __half2float(scale[rb]);
        if (direct) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int n0 = nt * 8 + 2 * q;
                if (n0 < n_tok)     { dst[(int64_t) n0 * sy + ra]       = c[nt][0] * sc0; dst[(int64_t) n0 * sy + rb]       = c[nt][2] * sc1; }
                if (n0 + 1 < n_tok) { dst[(int64_t) (n0 + 1) * sy + ra] = c[nt][1] * sc0; dst[(int64_t) (n0 + 1) * sy + rb] = c[nt][3] * sc1; }
            }
            continue;
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = nt * 8 + 2 * q;
            if (n0 < n_tok)     { atomicAdd(acc + n0 * h.rows + ra, c[nt][0] * sc0);       atomicAdd(acc + n0 * h.rows + rb, c[nt][2] * sc1); }
            if (n0 + 1 < n_tok) { atomicAdd(acc + (n0 + 1) * h.rows + ra, c[nt][1] * sc0); atomicAdd(acc + (n0 + 1) * h.rows + rb, c[nt][3] * sc1); }
        }
        if (ext_fin) {
            continue;
        }
        __threadfence();
        __syncthreads();
        const int grp = rt / 4;
        if (threadIdx.x == 0) {
            s_last = atomicAdd(cnt + grp, 1) == 4 * n_kc - 1;
        }
        __syncthreads();
        if (!s_last) {
            continue;
        }
        // last tile of this 512-row group: output rotation, signs, row map; clear the group for the next call.
        // One warp per token: lane l holds rows 16 l .. 16 l + 15; 4 butterfly stages in registers, 5 with shuffles.
        __threadfence();
        const uint8_t * sout = base + h.sout_off;
        for (int n = warp; n < n_tok; n += 8) {
            float4 * a4 = (float4 *) (acc + (int64_t) n * h.rows + grp * 512 + lane * 16);
            float v[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 t = __ldcg(a4 + i);
                v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
                __stcg(a4 + i, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
            if (h.flags & 1) {
#pragma unroll
                for (int hh = 1; hh < 16; hh <<= 1) {
#pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        if (!(e & hh)) {
                            const float x = v[e], y = v[e | hh];
                            v[e] = x + y; v[e | hh] = x - y;
                        }
                    }
                }
#pragma unroll
                for (int hl = 1; hl < 32; hl <<= 1) {
#pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        const float o = __shfl_xor_sync(0xFFFFFFFFu, v[e], hl);
                        v[e] = (lane & hl) ? o - v[e] : v[e] + o;
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int r = grp * 512 + lane * 16 + e;
                const float y = (h.flags & 1) ? v[e] * 0.044194173824159216f * ytr_sign(sout, r) : v[e];
                dst[(int64_t) n * sy + (rinv ? rinv[r] : r)] = y;
            }
        }
        if (threadIdx.x == 0) {
            cnt[grp] = 0;
        }
    }
}

// ---- int8 path ---------------------------------------------------------------------------------------------------------------

static __device__ __forceinline__ void ytr_mma8(int * d, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// four int8 weights of two steps from the extended stream e (the 12 - K bits before the quarter below bit 0, so the state of
// local step i is bits [K i, K i + 12)): one window at compile-time bit POS = K i holds both states; byte offsets into the table
template <int K, int POS>
static __device__ __forceinline__ uint32_t ytr_pairw(const uint32_t * e, const uint16_t * s_lut) {
    constexpr int word = POS >> 5, sh = POS & 31;
    uint32_t v;   // bit POS of e at bit 1
    if constexpr (sh == 0) {
        v = e[word] << 1;
    } else if constexpr (sh + 12 + K <= 32) {
        v = e[word] >> (sh - 1);
    } else {
        v = __funnelshift_r(e[word], e[word + 1], sh - 1);
    }
    const uint32_t lo = *(const uint16_t *) ((const char *) s_lut + (v & 0x1FFEu));
    const uint32_t hi = *(const uint16_t *) ((const char *) s_lut + ((v >> K) & 0x1FFEu));
    return __byte_perm(lo, hi, 0x5410);
}

// MMA steps M .. 7 of a 256-column block: step M takes local steps 4M .. 4M + 3 of rows g (ea) and g + 8 (eb), and the inputs at
// 8M .. 8M + 7 of the lane's quarter (xs + n xn, n = 8 nt + g)
template <int K, int NT, int M>
struct ytr_msteps {
    static __device__ __forceinline__ void run(const uint32_t * ea, const uint32_t * eb, const uint16_t * s_lut, const int8_t * xs,
                                               const int xn, const int n_tok, const int gi, int (*d)[4], uint4 * xb) {
        uint32_t a[4];
        a[0] = ytr_pairw<K, K * 4 * M>(ea, s_lut);
        a[1] = ytr_pairw<K, K * 4 * M>(eb, s_lut);
        a[2] = ytr_pairw<K, K * (4 * M + 2)>(ea, s_lut);
        a[3] = ytr_pairw<K, K * (4 * M + 2)>(eb, s_lut);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if constexpr (M % 2 == 0) {
                const int n = 8 * nt + gi;
                const uint4 * p = (const uint4 *) (xs + n * xn) + M / 2;
                xb[nt] = n < n_tok ? *p : make_uint4(0u, 0u, 0u, 0u);
                ytr_mma8(d[nt], a, xb[nt].x, xb[nt].y);
            } else {
                ytr_mma8(d[nt], a, xb[nt].z, xb[nt].w);
            }
        }
        ytr_msteps<K, NT, M + 1>::run(ea, eb, s_lut, xs, xn, n_tok, gi, d, xb);
    }
};
template <int K, int NT>
struct ytr_msteps<K, NT, 8> {
    static __device__ __forceinline__ void run(const uint32_t *, const uint32_t *, const uint16_t *, const int8_t *, const int, const int,
                                               const int, int (*)[4], uint4 *) {}
};

// L2 prefetch of share `part` of `parts` of [p, p + bytes)
static __device__ __forceinline__ void ytr_l2pf(const char * p, const uint32_t bytes, const uint32_t part, const uint32_t parts) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (p == nullptr || bytes == 0) {
        return;
    }
    const uint64_t a = ((uint64_t) bytes * part / parts) & ~15ull, b = ((uint64_t) bytes * (part + 1) / parts) & ~15ull;
    for (uint64_t o = a; o < b; o += 65536) {
        const uint32_t len = (uint32_t) (b - o < 65536 ? b - o : 65536);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(len) : "memory");
    }
#else
    GGML_UNUSED(p); GGML_UNUSED(bytes); GGML_UNUSED(part); GGML_UNUSED(parts);
#endif
}

// Walsh-Hadamard transform of 512 values held by a warp (unnormalized). Lane l holds positions l + 32 e, e = 0..15 (the stages
// commute, so the same code also serves lane l holding 16 l + e as in ytr_mm_kernel)
static __device__ __forceinline__ void ytr_wfwht(float (&v)[16], const int lane) {
#pragma unroll
    for (int hh = 1; hh < 16; hh <<= 1) {
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            if (!(e & hh)) {
                const float x = v[e], y = v[e | hh];
                v[e] = x + y; v[e | hh] = x - y;
            }
        }
    }
#pragma unroll
    for (int hl = 1; hl < 32; hl <<= 1) {
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const float o = __shfl_xor_sync(0xFFFFFFFFu, v[e], hl);
            v[e] = (lane & hl) ? o - v[e] : v[e] + o;
        }
    }
}

// normalize a transformed block, then int8 with one scale for its 512 values: q[l + 32 e], the scale at al
static __device__ __forceinline__ void ytr_wquant(float (&v)[16], const int lane, int8_t * q, float * al) {
    float m = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        v[e] *= YTR_HS;
        m = fmaxf(m, fabsf(v[e]));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    }
    const float a = m / 127.0f;
    if (lane == 0) {
        *al = a;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        int t = a > 0.0f ? __float2int_rn(v[e] / a) : 0;
        q[lane + 32 * e] = (int8_t) max(-127, min(127, t));
    }
}

// sign mask of positions c0 + l + 32 e (c0 a multiple of 512): bit e = bit l of word e of the block's bits (0 without bits).
// Weights only, so it can be read before the PDL wait.
static __device__ __forceinline__ uint32_t ytr_smask(const uint8_t * bits, const int c0, const int lane) {
    if (bits == nullptr) {
        return 0u;
    }
    const uint4 * w4 = (const uint4 *) (bits + c0 / 8);
    uint32_t m = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 w = __ldg(w4 + i);
        m |= ((w.x >> lane) & 1u) << (4 * i) | ((w.y >> lane) & 1u) << (4 * i + 1) | ((w.z >> lane) & 1u) << (4 * i + 2) |
             ((w.w >> lane) & 1u) << (4 * i + 3);
    }
    return m;
}

static __device__ __forceinline__ void ytr_apply_signs(float (&v)[16], const uint32_t m) {
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        if ((m >> e) & 1u) {
            v[e] = -v[e];
        }
    }
}

// partial sums p[l + 32 e] of a block, cleared for the next launch (unless keep)
static __device__ __forceinline__ void ytr_take16(float (&v)[16], float * p, const int lane, const bool keep = false) {
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        v[e] = p[lane + 32 * e];
    }
    if (keep) {
        return;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        p[lane + 32 * e] = 0.0f;
    }
}

// output rotation of a block of partial sums: H, 1/sqrt(512), signs (mask m)
static __device__ __forceinline__ void ytr_unrotate16(float (&v)[16], const int lane, const bool rot, const uint32_t m) {
    if (!rot) {
        return;
    }
    ytr_wfwht(v, lane);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        v[e] = ((m >> e) & 1u) ? -v[e] * YTR_HS : v[e] * YTR_HS;
    }
}

// rotated int8 input: x8[n][c] = int8 of (H S_in x_code)[c] / sqrt(512), x_code[c] = x[n][cmap[c]], alpha[n][c / 512] its scale.
// grid (ceil(cols / 2048), N), 4 warps, one block of 512 per warp
static __global__ void ytr_in8_kernel(const float * x, const int64_t sx, const uint8_t * sin, const int32_t * cmap, const int cols,
                                      int8_t * x8, float * alpha, const char * pf, const uint32_t pf_bytes) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, G = cols / 512, b = blockIdx.x * 4 + warp, n = blockIdx.y;
    ytr_l2pf(pf, pf_bytes, (n * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x, gridDim.x * gridDim.y * blockDim.x);
    const int      c0 = min(b, G - 1) * 512;
    const uint32_t sm = ytr_smask(sin, c0, lane);
    int ci[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int c = c0 + lane + 32 * e;
        ci[e] = cmap ? __ldg(cmap + c) : c;
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (b >= G) {
        return;
    }
    float v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        v[e] = x[n * sx + ci[e]];
    }
    ytr_apply_signs(v, sm);
    ytr_wfwht(v, lane);
    ytr_wquant(v, lane, x8 + (int64_t) n * cols + c0, alpha + (int64_t) n * G + b);
}

struct ytr_seg {
    const uint4 *    codes;     // lane layout
    const half *     scale;
    const uint16_t * lut8;
    float *          out;       // partial sums [n][rows] (added), or the result (direct: one worker, no output rotation)
    int64_t          so;        // stride of out per input row
    float            gamma;
    int              rows, cta_begin, nw, direct;
};

struct ytr_gv_args {
    ytr_seg        seg[YTR_MAX_SEG];
    const int8_t * x8;          // [n][cols]
    const float *  alpha;       // [n][cols / 512]
    int            n_seg, n_tok, cols, sg;   // sg: groups staged at a time
    // side rows: dense products of the F32 input xf, one block per row from block side_begin on (see ytr_side_block)
    const float *  xf;
    int64_t        sx;
    const char *   side_w[GGML_CUDA_YTR_MAX_SIDE];
    float *        side_out[GGML_CUDA_YTR_MAX_SIDE];
    const float *  side_pre[GGML_CUDA_YTR_MAX_SIDE];
    const float *  side_mul[GGML_CUDA_YTR_MAX_SIDE];
    int64_t        side_wrow[GGML_CUDA_YTR_MAX_SIDE], side_col[GGML_CUDA_YTR_MAX_SIDE];
    int            side_end[GGML_CUDA_YTR_MAX_SIDE], side_type[GGML_CUDA_YTR_MAX_SIDE], side_post[GGML_CUDA_YTR_MAX_SIDE];
    int            n_side, side_begin;
};

// a side row: block x - side_begin is row r of the side products, block y takes the input rows y, y + gridDim.y, ...
// out[t][r] = post(sum_k w[r][k] xf[t][k]). Loads are issued four at a time before use (a block of the main launch has few registers).
static __device__ __noinline__ void ytr_side_block(const ytr_gv_args & A, float * part) {
    const int row = (int) blockIdx.x - A.side_begin, nthr = blockDim.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int s = 0;
    while (s + 1 < A.n_side && row >= A.side_end[s]) {
        ++s;
    }
    const int    lr   = row - (s > 0 ? A.side_end[s - 1] : 0);
    const char * wrow = A.side_w[s] + (size_t) lr * A.side_wrow[s];
    const int    wt   = A.side_type[s];
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (row >= A.side_end[A.n_side - 1]) {
        return;
    }
    for (int t = blockIdx.y; t < A.n_tok; t += gridDim.y) {
        const float * x = A.xf + (size_t) t * A.sx;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < A.cols; i += 4 * nthr) {
            float w[4], v[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int k = i + l * nthr;
                const bool in = k < A.cols;
                const int  kk = in ? k : i;
                w[l] = !in ? 0.0f : wt == 0 ? ((const float *) wrow)[kk] : wt == 1 ? __half2float(((const half *) wrow)[kk]) :
                       __uint_as_float((uint32_t) ((const uint16_t *) wrow)[kk] << 16);
                v[l] = x[kk];
            }
            acc += (w[0] * v[0] + w[1] * v[1]) + (w[2] * v[2] + w[3] * v[3]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        }
        if (lane == 0) {
            part[warp] = acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int k = 0; k < nthr / 32; ++k) {
                sum += part[k];
            }
            if (A.side_post[s] == 1) {
                sum = 1.0f / (1.0f + expf(-sum));
            } else if (A.side_post[s] == 2) {
                sum = sum + A.side_pre[s][lr];
                sum = (sum > 20.0f) ? sum : logf(1.0f + expf(sum));
                sum = sum * A.side_mul[s][lr];
            }
            A.side_out[s][(size_t) t * A.side_col[s] + lr] = sum;
        }
        __syncthreads();
    }
}

// up to 4 products of the same input: blocks of UW warps (one 16-row tile each) on x, split-K workers on y; worker w takes the
// 512-column groups w per .. w per + per - 1 (per = ceil(G / nw)). The block stages the inputs and scales of A.sg groups at a time
// in shared memory (dynamic: sg x n_tok x YTR_XROW bytes + sg x n_tok floats); the next code block is loaded while the current one is
// decoded. All warps of a block share the worker, so they have the same number of blocks (rows are a multiple of 128).
template <int K, int NT, int UW, int MINB>
static __global__ void __launch_bounds__(UW * 32, MINB) ytr_gv8_kernel(const ytr_gv_args A) {
    constexpr int      NU = K / 2, T = 12 - K;
    constexpr uint32_t TMASK = (1u << T) - 1u;
    __shared__ __align__(16) uint16_t s_lut[4096];
    extern __shared__ __align__(16) int8_t s_x[];

    if (A.n_side > 0 && (int) blockIdx.x >= A.side_begin) {
        ytr_side_block(A, (float *) s_lut);
        return;
    }
    int si = 0;
    while (si + 1 < A.n_seg && (int) blockIdx.x >= A.seg[si + 1].cta_begin) {
        ++si;
    }
    const ytr_seg & S = A.seg[si];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, tig = lane % 4;
    const int worker = blockIdx.y, nw = S.nw, G = A.cols / 512, seqs = A.cols / 256;
    const int tile = ((int) blockIdx.x - S.cta_begin) * UW + warp;
    const int per  = (G + nw - 1) / nw, g0 = worker * per;   // worker w: groups g0 .. g0 + ngw - 1 (contiguous codes)
    const int ngw  = worker < nw ? min(per, G - g0) : 0;
    const int nblk = tile < S.rows / 16 ? 2 * ngw : 0;
    const uint4 * sp = S.codes + (size_t) (nblk ? tile : 0) * seqs * (NU * 32) + lane;

    // before the wait: the first code block and the code table (no earlier kernel writes them)
    uint4 nx[NU];
    if (nblk) {
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            nx[u] = __ldg(sp + (size_t) (2 * g0) * (NU * 32) + 32 * u);
        }
    }
    for (int i = threadIdx.x; i < 512; i += UW * 32) {
        ((uint4 *) s_lut)[i] = __ldg((const uint4 *) S.lut8 + i);
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    __syncthreads();
    if (!nblk) {
        return;
    }
    float * s_al = (float *) (s_x + (size_t) A.sg * A.n_tok * YTR_XROW);

    const int prev = (lane & ~3) | ((tig + 3) & 3);   // lane holding the previous quarter of the same rows
    float c[NT][4];
    int   d[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[nt][k] = 0.0f;
            d[nt][k] = 0;
        }
    }
    for (int bi = 0; bi < nblk; ++bi) {
        const int g = g0 + (bi >> 1), ig = (bi >> 1) % A.sg;
        if (bi % (2 * A.sg) == 0) {   // stage the next sg groups: s_x[(ig n_tok + n) 512 + c], their scales at s_al[ig n_tok + n]
            const int ng = min(A.sg, ngw - (bi >> 1));
            __syncthreads();
            for (int k = threadIdx.x; k < ng * A.n_tok * 32; k += UW * 32) {
                const int i = k / (A.n_tok * 32), n = (k / 32) % A.n_tok, c = k % 32;
                *(uint4 *) (s_x + (size_t) (i * A.n_tok + n) * YTR_XROW + 16 * c) =
                    __ldcg((const uint4 *) (A.x8 + (size_t) n * A.cols + (size_t) (g + i) * 512) + c);
            }
            for (int k = threadIdx.x; k < ng * A.n_tok; k += UW * 32) {
                s_al[k] = __ldcg(A.alpha + (size_t) (k % A.n_tok) * G + g + k / A.n_tok);
            }
            __syncthreads();
        }
        uint32_t w[2 * K];
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            w[4 * u] = nx[u].x; w[4 * u + 1] = nx[u].y; w[4 * u + 2] = nx[u].z; w[4 * u + 3] = nx[u].w;
        }
        if (bi + 1 < nblk) {
            const int s1 = 2 * g0 + bi + 1;
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                nx[u] = __ldg(sp + (size_t) s1 * (NU * 32) + 32 * u);
            }
        }
        // the T bits before this quarter: the end of the previous lane's quarter (zeros at the start of the sequence)
        const uint32_t e   = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);
        const uint32_t nbv = __shfl_sync(0xFFFFFFFFu, e, prev);
        uint32_t ea[K + 1], eb[K + 1];
        ea[0] = (tig ? (nbv & TMASK) : 0u) | (w[0] << T);
        eb[0] = (tig ? (nbv >> T) : 0u) | (w[K] << T);
#pragma unroll
        for (int k = 1; k < K; ++k) {
            ea[k] = __funnelshift_l(w[k - 1], w[k], T);
            eb[k] = __funnelshift_l(w[K + k - 1], w[K + k], T);
        }
        ea[K] = w[K - 1] >> (32 - T);
        eb[K] = w[2 * K - 1] >> (32 - T);
        const int8_t * xs = s_x + (size_t) ig * A.n_tok * YTR_XROW + (bi & 1) * 256 + 64 * tig;
        uint4 xb[NT];
        ytr_msteps<K, NT, 0>::run(ea, eb, s_lut, xs, YTR_XROW, A.n_tok, gi, d, xb);
        if (bi & 1) {   // end of a 512-column group: its input scales
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int   t0 = 8 * nt + 2 * tig;
                const float * ap = s_al + ig * A.n_tok;
                const float a0 = t0 < A.n_tok ? ap[t0] : 0.0f;
                const float a1 = t0 + 1 < A.n_tok ? ap[t0 + 1] : 0.0f;
                c[nt][0] += a0 * (float) d[nt][0]; c[nt][2] += a0 * (float) d[nt][2];
                c[nt][1] += a1 * (float) d[nt][1]; c[nt][3] += a1 * (float) d[nt][3];
                d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0;
            }
        }
    }
    const int   ra  = tile * 16 + gi, rb = ra + 8;
    const float sc0 = __half2float(S.scale[ra]) * S.gamma, sc1 = __half2float(S.scale[rb]) * S.gamma;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int n0 = 8 * nt + 2 * tig;
        float * o0 = S.out + (int64_t) n0 * S.so;
        float * o1 = o0 + S.so;
        if (S.direct) {
            if (n0 < A.n_tok)     { o0[ra] = c[nt][0] * sc0; o0[rb] = c[nt][2] * sc1; }
            if (n0 + 1 < A.n_tok) { o1[ra] = c[nt][1] * sc0; o1[rb] = c[nt][3] * sc1; }
        } else {
            if (n0 < A.n_tok)     { atomicAdd(o0 + ra, c[nt][0] * sc0); atomicAdd(o0 + rb, c[nt][2] * sc1); }
            if (n0 + 1 < A.n_tok) { atomicAdd(o1 + ra, c[nt][1] * sc0); atomicAdd(o1 + rb, c[nt][3] * sc1); }
        }
    }
}

static __device__ __forceinline__ void ytr_cp16(void * smem, const void * gmem, const bool ok) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}

static __device__ __forceinline__ void ytr_ldsm4(uint32_t * r, const void * smem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// local steps 4 M .. 4 MEND - 1 of rows g (ea -> oa) and g + 8 (eb -> ob): two MMA-ready int8 words per step pair, oa[2 (M % 4)] ..
template <int K, int M, int MEND>
struct ytr_hsteps {
    static __device__ __forceinline__ void run(const uint32_t * ea, const uint32_t * eb, const uint16_t * s_lut, uint32_t * oa, uint32_t * ob) {
        oa[2 * (M % 4)]     = ytr_pairw<K, K * 4 * M>(ea, s_lut);
        ob[2 * (M % 4)]     = ytr_pairw<K, K * 4 * M>(eb, s_lut);
        oa[2 * (M % 4) + 1] = ytr_pairw<K, K * (4 * M + 2)>(ea, s_lut);
        ob[2 * (M % 4) + 1] = ytr_pairw<K, K * (4 * M + 2)>(eb, s_lut);
        ytr_hsteps<K, M + 1, MEND>::run(ea, eb, s_lut, oa, ob);
    }
};
template <int K, int MEND>
struct ytr_hsteps<K, MEND, MEND> {
    static __device__ __forceinline__ void run(const uint32_t *, const uint32_t *, const uint16_t *, uint32_t *, uint32_t *) {}
};

static constexpr int YTR_GW_AROW = 128;   // bytes per decoded row of a half block (16-byte chunks swizzled by row % 8)
static constexpr int YTR_GW_BROW = 144;   // bytes per staged input row of a half block (128 + 16: ldmatrix of 8 rows is conflict free)

// half H of 256-column block s (columns 64 q + 32 H .. + 31 of each quarter q): stage the inputs, decode the warp's 16 rows into its
// shared tile, then the MMAs of all token tiles. A tile row and an input row hold the same columns in the same order.
template <int K, int NTT, int H>
static __device__ __forceinline__ void ytr_gw_half(const ytr_gv_args & A, const int s, const int g, const bool group_start,
        const uint32_t * ea, const uint32_t * eb, const uint16_t * s_lut, int8_t * aw, int8_t * s_b, float * s_al, int (*d)[4]) {
    constexpr int NR = NTT * 8;
    const int lane = threadIdx.x % 32, gi = lane / 4, tig = lane % 4, G = A.cols / 512;
    __syncthreads();   // the previous half's inputs and tiles are consumed
    for (int k = threadIdx.x; k < NR * 8; k += 256) {
        const int n = k / 8, q = (k / 2) % 4, p = k % 2;
        *(uint4 *) (s_b + n * YTR_GW_BROW + 32 * q + 16 * p) = n < A.n_tok ?
            __ldcg((const uint4 *) (A.x8 + (size_t) n * A.cols + s * 256 + 64 * q + 32 * H + 16 * p)) : make_uint4(0u, 0u, 0u, 0u);
    }
    if (H == 0 && group_start) {
        for (int k = threadIdx.x; k < NR; k += 256) {
            s_al[k] = k < A.n_tok ? __ldcg(A.alpha + (size_t) k * G + g) : 0.0f;
        }
    }
    uint32_t oa[8], ob[8];
    ytr_hsteps<K, 4 * H, 4 * H + 4>::run(ea, eb, s_lut, oa, ob);
    const int ra = gi, rb = gi + 8;
    *(uint4 *) (aw + ra * YTR_GW_AROW + (((2 * tig)     ^ (ra & 7)) << 4)) = make_uint4(oa[0], oa[1], oa[2], oa[3]);
    *(uint4 *) (aw + ra * YTR_GW_AROW + (((2 * tig + 1) ^ (ra & 7)) << 4)) = make_uint4(oa[4], oa[5], oa[6], oa[7]);
    *(uint4 *) (aw + rb * YTR_GW_AROW + (((2 * tig)     ^ (rb & 7)) << 4)) = make_uint4(ob[0], ob[1], ob[2], ob[3]);
    *(uint4 *) (aw + rb * YTR_GW_AROW + (((2 * tig + 1) ^ (rb & 7)) << 4)) = make_uint4(ob[4], ob[5], ob[6], ob[7]);
    __syncthreads();
#pragma unroll
    for (int st = 0; st < 4; ++st) {   // k step st = the 32 bytes of quarter st
        uint32_t a[4];
        {
            const int r = (lane % 8) + 8 * ((lane / 8) % 2), ch = 2 * st + lane / 16;
            ytr_ldsm4(a, aw + r * YTR_GW_AROW + ((ch ^ (r & 7)) << 4));
        }
#pragma unroll
        for (int np = 0; np < NTT / 2; ++np) {
            const int r = 16 * np + (lane % 8) + 8 * (lane / 16), ch = 2 * st + (lane / 8) % 2;
            uint32_t b[4];
            ytr_ldsm4(b, s_b + r * YTR_GW_BROW + ch * 16);
            ytr_mma8(d[2 * np], a, b[0], b[1]);
            ytr_mma8(d[2 * np + 1], a, b[2], b[3]);
        }
    }
}

// 17 to 64 input rows: same launch, split-K workers and outputs as ytr_gv8_kernel (8 warps of 16 rows), but each warp decodes its
// rows into shared memory and the MMAs read both operands with ldmatrix, so the accumulators of all token tiles fit without the
// decode state spilling. Dynamic shared memory: 8 x 16 x YTR_GW_AROW + NTT 8 x (YTR_GW_BROW + 4) bytes.
template <int K, int NTT>
static __global__ void __launch_bounds__(256, 2) ytr_gw8_kernel(const ytr_gv_args A) {
    constexpr int      NU = K / 2, T = 12 - K, UW = 8, NR = NTT * 8;
    constexpr uint32_t TMASK = (1u << T) - 1u;
    __shared__ __align__(16) uint16_t s_lut[4096];
    extern __shared__ __align__(16) int8_t s_dyn[];
    int8_t * s_a  = s_dyn;
    int8_t * s_b  = s_dyn + UW * 16 * YTR_GW_AROW;
    float *  s_al = (float *) (s_b + NR * YTR_GW_BROW);

    int si = 0;
    while (si + 1 < A.n_seg && (int) blockIdx.x >= A.seg[si + 1].cta_begin) {
        ++si;
    }
    const ytr_seg & S = A.seg[si];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, tig = lane % 4;
    const int worker = blockIdx.y, nw = S.nw, G = A.cols / 512, seqs = A.cols / 256;
    const int tile = ((int) blockIdx.x - S.cta_begin) * UW + warp;
    const int per  = (G + nw - 1) / nw, g0 = worker * per;
    const int ngw  = worker < nw ? min(per, G - g0) : 0;
    const int nblk = tile < S.rows / 16 ? 2 * ngw : 0;
    const uint4 * sp = S.codes + (size_t) (nblk ? tile : 0) * seqs * (NU * 32) + lane;

    uint4 nx[NU];
    if (nblk) {
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            nx[u] = __ldg(sp + (size_t) (2 * g0) * (NU * 32) + 32 * u);
        }
    }
    for (int i = threadIdx.x; i < 512; i += UW * 32) {
        ((uint4 *) s_lut)[i] = __ldg((const uint4 *) S.lut8 + i);
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    __syncthreads();
    if (!nblk) {
        return;
    }
    int8_t * aw = s_a + warp * 16 * YTR_GW_AROW;
    const int prev = (lane & ~3) | ((tig + 3) & 3);
    float c[NTT][4];
    int   d[NTT][4];
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[nt][k] = 0.0f;
            d[nt][k] = 0;
        }
    }
    for (int bi = 0; bi < nblk; ++bi) {
        const int g = g0 + (bi >> 1), s = 2 * g0 + bi;
        uint32_t w[2 * K];
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            w[4 * u] = nx[u].x; w[4 * u + 1] = nx[u].y; w[4 * u + 2] = nx[u].z; w[4 * u + 3] = nx[u].w;
        }
        if (bi + 1 < nblk) {
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                nx[u] = __ldg(sp + (size_t) (s + 1) * (NU * 32) + 32 * u);
            }
        }
        const uint32_t e   = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);
        const uint32_t nbv = __shfl_sync(0xFFFFFFFFu, e, prev);
        uint32_t ea[K + 1], eb[K + 1];
        ea[0] = (tig ? (nbv & TMASK) : 0u) | (w[0] << T);
        eb[0] = (tig ? (nbv >> T) : 0u) | (w[K] << T);
#pragma unroll
        for (int k = 1; k < K; ++k) {
            ea[k] = __funnelshift_l(w[k - 1], w[k], T);
            eb[k] = __funnelshift_l(w[K + k - 1], w[K + k], T);
        }
        ea[K] = w[K - 1] >> (32 - T);
        eb[K] = w[2 * K - 1] >> (32 - T);
        ytr_gw_half<K, NTT, 0>(A, s, g, (bi & 1) == 0, ea, eb, s_lut, aw, s_b, s_al, d);
        ytr_gw_half<K, NTT, 1>(A, s, g, false, ea, eb, s_lut, aw, s_b, s_al, d);
        if (bi & 1) {   // end of a 512-column group: its input scales
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt) {
                const int   t0 = 8 * nt + 2 * tig;
                const float a0 = s_al[t0], a1 = s_al[t0 + 1];
                c[nt][0] += a0 * (float) d[nt][0]; c[nt][2] += a0 * (float) d[nt][2];
                c[nt][1] += a1 * (float) d[nt][1]; c[nt][3] += a1 * (float) d[nt][3];
                d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0;
            }
        }
    }
    const int   ra  = tile * 16 + gi, rb = ra + 8;
    const float sc0 = __half2float(S.scale[ra]) * S.gamma, sc1 = __half2float(S.scale[rb]) * S.gamma;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
        const int n0 = 8 * nt + 2 * tig;
        float * o0 = S.out + (int64_t) n0 * S.so;
        float * o1 = o0 + S.so;
        if (S.direct) {
            if (n0 < A.n_tok)     { o0[ra] = c[nt][0] * sc0; o0[rb] = c[nt][2] * sc1; }
            if (n0 + 1 < A.n_tok) { o1[ra] = c[nt][1] * sc0; o1[rb] = c[nt][3] * sc1; }
        } else {
            if (n0 < A.n_tok)     { atomicAdd(o0 + ra, c[nt][0] * sc0); atomicAdd(o0 + rb, c[nt][2] * sc1); }
            if (n0 + 1 < A.n_tok) { atomicAdd(o1 + ra, c[nt][1] * sc0); atomicAdd(o1 + rb, c[nt][3] * sc1); }
        }
    }
}

// ---- finishing kernels: output rotation, signs and row map of the summed partials (cleared for the next launch) -------------

struct ytr_fin_seg {
    const uint8_t * sout;       // output signs, null without output rotation
    const int *     rinv;       // row map or null
    float *         acc;        // [n][rows]
    float *         dst;
    int64_t         sy;
    int             rows, grp_begin;
};

struct ytr_fin_args {
    ytr_fin_seg  seg[YTR_MAX_SEG];
    const char * pf;
    int          n_seg, n_tok, keep;   // keep: do not clear the partial sums (a one-off buffer)
    uint32_t     pf_bytes;
};

// grid (sum of rows / 512, ceil(N / 8)), 8 warps, one token per warp
static __global__ void ytr_fin_kernel(const ytr_fin_args A) {
    int si = 0;
    while (si + 1 < A.n_seg && (int) blockIdx.x >= A.seg[si + 1].grp_begin) {
        ++si;
    }
    const ytr_fin_seg & S = A.seg[si];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = blockIdx.y * 8 + warp;
    ytr_l2pf(A.pf, A.pf_bytes, (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x, gridDim.x * gridDim.y * blockDim.x);
    const int      r0 = ((int) blockIdx.x - S.grp_begin) * 512;
    const uint32_t sm = ytr_smask(S.sout, r0, lane);
    int ri[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int r = r0 + lane + 32 * e;
        ri[e] = S.rinv ? __ldg(S.rinv + r) : r;
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (n >= A.n_tok) {
        return;
    }
    float v[16];
    ytr_take16(v, S.acc + (int64_t) n * S.rows + r0, lane, A.keep != 0);
    ytr_unrotate16(v, lane, S.sout != nullptr, sm);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        S.dst[n * S.sy + ri[e]] = v[e];
    }
}

// E1: finish of a product of D rows, then ADD (residual), RMS_NORM, MUL (weight) and the next product's int8 input.
struct ytr_e1_args {
    float *         acc;        // partial sums of the product [n][D]
    const uint8_t * sout;
    const int *     rinv;
    float *         mm_out;     // the product
    const float *   res;        // other operand of the ADD
    float *         add_out;
    float *         norm_out;   // RMS_NORM result
    const float *   w;          // MUL weight [D]
    float *         mul_out;
    const uint8_t * sin;        // input signs of the next product, null: no int8 input
    const int32_t * cmap;
    int8_t *        x8;
    float *         alpha;
    const char *    pf;
    int64_t         s_mm, s_res, s_add, s_norm, s_mul;
    float           eps;
    int             D;
    uint32_t        pf_bytes;
};

// grid N, D / 512 warps (one block of 512 rows each); shared memory D + 32 floats. Weights (signs, maps, norm weight) are read
// before the PDL wait.
static __global__ void ytr_e1_kernel(const ytr_e1_args A) {
    extern __shared__ float s_y[];
    float * s_red = s_y + A.D;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = blockIdx.x, nb = A.D / 512, c0 = warp * 512;
    ytr_l2pf(A.pf, A.pf_bytes, n * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    const uint32_t so = ytr_smask(A.sout, c0, lane), si = ytr_smask(A.sin, c0, lane);
    float wv[16];
    int   ri[16], ci[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int r = c0 + lane + 32 * e;
        wv[e] = __ldg(A.w + r);
        ri[e] = A.rinv ? __ldg(A.rinv + r) : r;
        ci[e] = A.cmap ? __ldg(A.cmap + r) : r;
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    float v[16], rs[16];
    ytr_take16(v, A.acc + (int64_t) n * A.D + c0, lane);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        rs[e] = A.res[n * A.s_res + c0 + lane + 32 * e];
    }
    ytr_unrotate16(v, lane, A.sout != nullptr, so);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        s_y[ri[e]] = v[e];
    }
    __syncthreads();
    float ss = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int   r = c0 + lane + 32 * e;
        const float m = s_y[r];
        if (A.mm_out) {
            A.mm_out[n * A.s_mm + r] = m;
        }
        const float h = m + rs[e];
        A.add_out[n * A.s_add + r] = h;
        v[e] = h;
        ss += h * h;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
    }
    if (lane == 0) {
        s_red[warp] = ss;
    }
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < nb; ++k) {
        tot += s_red[k];
    }
    const float scale = rsqrtf(tot / A.D + A.eps);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int   r  = c0 + lane + 32 * e;
        const float nv = v[e] * scale;
        if (A.norm_out) {
            A.norm_out[n * A.s_norm + r] = nv;
        }
        v[e] = nv * wv[e];
        A.mul_out[n * A.s_mul + r] = v[e];
    }
    if (A.x8 == nullptr) {
        return;
    }
    if (A.cmap) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            s_y[c0 + lane + 32 * e] = v[e];
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            v[e] = s_y[ci[e]];
        }
    }
    ytr_apply_signs(v, si);
    ytr_wfwht(v, lane);
    ytr_wquant(v, lane, A.x8 + (int64_t) n * A.D + c0, A.alpha + (int64_t) n * nb + warp);
}

// E2: finish of gate and up (F rows each), SWIGLU, and the int8 input of the down product (no column map)
struct ytr_e2_args {
    float *         acc_g;
    float *         acc_u;
    const uint8_t * sout_g;
    const uint8_t * sout_u;
    float *         g_out;
    float *         u_out;
    float *         sw_out;
    const uint8_t * sin;        // input signs of the down product, null: no int8 input
    int8_t *        x8;
    float *         alpha;
    const char *    pf;
    int64_t         s_g, s_u, s_sw;
    int             F;
    uint32_t        pf_bytes;
};

// grid (ceil(F / 2048), N), 4 warps, one block of 512 per warp
static __global__ void ytr_e2_kernel(const ytr_e2_args A) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, b = blockIdx.x * 4 + warp, n = blockIdx.y, nb = A.F / 512;
    ytr_l2pf(A.pf, A.pf_bytes, (n * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x, gridDim.x * gridDim.y * blockDim.x);
    const int      c0 = min(b, nb - 1) * 512;
    const uint32_t sg = ytr_smask(A.sout_g, c0, lane), su = ytr_smask(A.sout_u, c0, lane), si = ytr_smask(A.sin, c0, lane);
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (b >= nb) {
        return;
    }
    float g[16], u[16];
    ytr_take16(g, A.acc_g + (int64_t) n * A.F + c0, lane);
    ytr_take16(u, A.acc_u + (int64_t) n * A.F + c0, lane);
    ytr_unrotate16(g, lane, A.sout_g != nullptr, sg);
    ytr_unrotate16(u, lane, A.sout_u != nullptr, su);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int r = c0 + lane + 32 * e;
        if (A.g_out) {
            A.g_out[n * A.s_g + r] = g[e];
        }
        if (A.u_out) {
            A.u_out[n * A.s_u + r] = u[e];
        }
        g[e] = ggml_cuda_op_silu_single(g[e]) * u[e];
        if (A.sw_out) {
            A.sw_out[n * A.s_sw + r] = g[e];
        }
    }
    if (A.x8 == nullptr) {
        return;
    }
    ytr_apply_signs(g, si);
    ytr_wfwht(g, lane);
    ytr_wquant(g, lane, A.x8 + (int64_t) n * A.F + c0, A.alpha + (int64_t) n * nb + b);
}

// ---- more than 64 input rows: weights decoded to int8 once per call, then an int8 GEMM -------------------------------------

// predecode steps M .. 7 of a 256-column block into the int8 words of rows g (oa) and g + 8 (ob), 8 bytes per step
template <int K, int M>
struct ytr_dsteps {
    static __device__ __forceinline__ void run(const uint32_t * ea, const uint32_t * eb, const uint16_t * s_lut, uint32_t * oa, uint32_t * ob) {
        oa[2 * M]     = ytr_pairw<K, K * 4 * M>(ea, s_lut);
        ob[2 * M]     = ytr_pairw<K, K * 4 * M>(eb, s_lut);
        oa[2 * M + 1] = ytr_pairw<K, K * (4 * M + 2)>(ea, s_lut);
        ob[2 * M + 1] = ytr_pairw<K, K * (4 * M + 2)>(eb, s_lut);
        ytr_dsteps<K, M + 1>::run(ea, eb, s_lut, oa, ob);
    }
};
template <int K>
struct ytr_dsteps<K, 8> {
    static __device__ __forceinline__ void run(const uint32_t *, const uint32_t *, const uint16_t *, uint32_t *, uint32_t *) {}
};

// W8[r - r0][c] = int8 code table value of Q[r][c] for rows r0 .. r0 + nr (lane layout); grid (seqs, nr / 128), 8 warps, one
// 16-row tile per warp
template <int K>
static __global__ void __launch_bounds__(256) ytr_predecode_kernel(const uint4 * codes, const uint16_t * lut8, const int cols, const int r0,
                                                                  int8_t * W8) {
    constexpr int      NU = K / 2, T = 12 - K;
    constexpr uint32_t TMASK = (1u << T) - 1u;
    __shared__ __align__(16) uint16_t s_lut[4096];
    for (int i = threadIdx.x; i < 512; i += 256) {
        ((uint4 *) s_lut)[i] = __ldg((const uint4 *) lut8 + i);
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, tig = lane % 4, seqs = cols / 256, s = blockIdx.x;
    const int tile = r0 / 16 + blockIdx.y * 8 + warp;
    const uint4 * sp = codes + ((size_t) tile * seqs + s) * (NU * 32) + lane;
    uint32_t w[2 * K];
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const uint4 v = __ldg(sp + 32 * u);
        w[4 * u] = v.x; w[4 * u + 1] = v.y; w[4 * u + 2] = v.z; w[4 * u + 3] = v.w;
    }
    ggml_cuda_pdl_sync();   // the workspace may still be read by the previous GEMM
    ggml_cuda_pdl_lc();
    __syncthreads();
    const int      prev = (lane & ~3) | ((tig + 3) & 3);
    const uint32_t e    = (w[K - 1] >> (32 - T)) | ((w[2 * K - 1] >> (32 - T)) << T);
    const uint32_t nbv  = __shfl_sync(0xFFFFFFFFu, e, prev);
    uint32_t ea[K + 1], eb[K + 1];
    ea[0] = (tig ? (nbv & TMASK) : 0u) | (w[0] << T);
    eb[0] = (tig ? (nbv >> T) : 0u) | (w[K] << T);
#pragma unroll
    for (int k = 1; k < K; ++k) {
        ea[k] = __funnelshift_l(w[k - 1], w[k], T);
        eb[k] = __funnelshift_l(w[K + k - 1], w[K + k], T);
    }
    ea[K] = w[K - 1] >> (32 - T);
    eb[K] = w[2 * K - 1] >> (32 - T);
    uint32_t oa[16], ob[16];
    ytr_dsteps<K, 0>::run(ea, eb, s_lut, oa, ob);
    const int ra = (tile * 16 + gi) - r0, rb = ra + 8;
    uint4 * pa = (uint4 *) (W8 + (size_t) ra * cols + s * 256 + 64 * tig);
    uint4 * pb = (uint4 *) (W8 + (size_t) rb * cols + s * 256 + 64 * tig);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa[i] = make_uint4(oa[4 * i], oa[4 * i + 1], oa[4 * i + 2], oa[4 * i + 3]);
        pb[i] = make_uint4(ob[4 * i], ob[4 * i + 1], ob[4 * i + 2], ob[4 * i + 3]);
    }
}

// 64-byte shared rows of 4 chunks of 16 bytes, chunk c of row r at c ^ ((r >> 1) & 3): ldmatrix of 8 rows is conflict free
static __device__ __forceinline__ int ytr_swz(const int r, const int c) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

// C[n][r] = scale[r] gamma sum_g alpha[n][g] sum_{k in g} W8[r][k] X8[n][k]: block tile 128 rows x 64 inputs, k step 64,
// 3-stage cp.async pipeline; 8 warps of 32 rows x 32 inputs; int32 sums per 512-column group. grid (rows / 128, ceil(N / 64))
static constexpr int YTR_GM_ST = 3;
static __global__ void __launch_bounds__(256, 2) ytr_gemm8_kernel(const int8_t * W8, const int8_t * X8, const float * alpha, const half * scale,
                                                                 const float gamma, float * C, const int64_t sc, const int N, const int cols) {
    __shared__ __align__(128) int8_t s_a[YTR_GM_ST][128 * 64];
    __shared__ __align__(128) int8_t s_b[YTR_GM_ST][64 * 64];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gi = lane / 4, tig = lane % 4, wm = warp / 2, wn = warp % 2;
    const int r0 = blockIdx.x * 128, n0 = blockIdx.y * 64, KT = cols / 64, G = cols / 512;
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    auto load = [&](const int st, const int kt) {
        const int k0 = kt * 64;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = threadIdx.x + 256 * i, r = q / 4, c = q % 4;
            ytr_cp16(s_a[st] + ytr_swz(r, c), W8 + (size_t) (r0 + r) * cols + k0 + 16 * c, true);
        }
        const int r = threadIdx.x / 4, c = threadIdx.x % 4, n = n0 + r;
        ytr_cp16(s_b[st] + ytr_swz(r, c), X8 + (size_t) min(n, N - 1) * cols + k0 + 16 * c, n < N);
    };
#pragma unroll
    for (int st = 0; st < YTR_GM_ST - 1; ++st) {
        if (st < KT) {
            load(st, st);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    float c[2][4][4];
    int   d[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                c[mi][ni][q] = 0.0f;
                d[mi][ni][q] = 0;
            }
        }
    }
    for (int kt = 0; kt < KT; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(YTR_GM_ST - 2));
        __syncthreads();
        const int st = kt % YTR_GM_ST;
        if (kt + YTR_GM_ST - 1 < KT) {
            load((kt + YTR_GM_ST - 1) % YTR_GM_ST, kt + YTR_GM_ST - 1);
        }
        asm volatile("cp.async.commit_group;\n" ::);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm * 32 + mi * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                ytr_ldsm4(a[mi], s_a[st] + ytr_swz(r, 2 * kk + lane / 16));
            }
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int r = wn * 32 + np * 16 + (lane % 8) + 8 * (lane / 16);
                uint32_t t[4];
                ytr_ldsm4(t, s_b[st] + ytr_swz(r, 2 * kk + (lane / 8) % 2));
                b[2 * np][0] = t[0]; b[2 * np][1] = t[1]; b[2 * np + 1][0] = t[2]; b[2 * np + 1][1] = t[3];
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    ytr_mma8(d[mi][ni], a[mi], b[ni][0], b[ni][1]);
                }
            }
        }
        if (kt % 8 == 7) {   // end of a 512-column group: its input scales
            const int g = kt / 8;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int   t0 = n0 + wn * 32 + ni * 8 + 2 * tig;
                const float a0 = t0 < N ? alpha[(size_t) t0 * G + g] : 0.0f;
                const float a1 = t0 + 1 < N ? alpha[(size_t) (t0 + 1) * G + g] : 0.0f;
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    c[mi][ni][0] += a0 * (float) d[mi][ni][0]; c[mi][ni][2] += a0 * (float) d[mi][ni][2];
                    c[mi][ni][1] += a1 * (float) d[mi][ni][1]; c[mi][ni][3] += a1 * (float) d[mi][ni][3];
                    d[mi][ni][0] = d[mi][ni][1] = d[mi][ni][2] = d[mi][ni][3] = 0;
                }
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int   ra  = r0 + wm * 32 + mi * 16 + gi, rb = ra + 8;
        const float sa  = __half2float(scale[ra]) * gamma, sb = __half2float(scale[rb]) * gamma;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int t0 = n0 + wn * 32 + ni * 8 + 2 * tig;
            if (t0 < N)     { C[t0 * sc + ra]       = c[mi][ni][0] * sa; C[t0 * sc + rb]       = c[mi][ni][2] * sb; }
            if (t0 + 1 < N) { C[(t0 + 1) * sc + ra] = c[mi][ni][1] * sa; C[(t0 + 1) * sc + rb] = c[mi][ni][3] * sb; }
        }
    }
}

// embedding lookup: dst[n] = row ids[n] of W = diag(scale) Q H S_in (row-major codes, no output rotation or maps).
// grid (ceil(cols / 2048), n), 4 warps, one block of 512 columns per warp, lane l holding columns l + 32 e
static __global__ void ytr_rows_kernel(const uint8_t * base, const ytr_header h, const int32_t * ids, const int64_t s_ids, float * dst,
                                       const int64_t s_dst) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, G = h.cols / 512, b = blockIdx.x * 4 + warp, n = blockIdx.y, K = h.bits;
    const int      c0 = min(b, G - 1) * 512;
    const uint32_t sm = ytr_smask(base + h.sin_off, c0, lane);
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (b >= G) {
        return;
    }
    const int64_t    row   = ids[n * s_ids];
    const uint32_t * cw    = (const uint32_t *) (base + h.code_off) + row * (h.cols / 256) * 4 * K;
    const half2 *    lut   = (const half2 *) (base + h.lut_off);
    const float      sc    = __half2float(((const half *) (base + h.scale_off))[row]);
    float v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int c = c0 + lane + 32 * e, s = c / 256, j = (c % 256) / 2, end = K * (j + 1);
        const uint32_t * w = cw + s * 4 * K;
        uint32_t st;
        if (end < 12) {
            st = (__ldg(w) & ((1u << end) - 1u)) << (12 - end);
        } else {
            const int start = end - 12, wi = start >> 5, sh = start & 31;
            st = __ldg(w + wi) >> sh;
            if (sh > 20) {
                st |= __ldg(w + wi + 1) << (32 - sh);
            }
            st &= 0xFFFu;
        }
        const float2 l = __half22float2(__ldg(lut + st));
        v[e] = ((c & 1) ? l.y : l.x) * sc;
    }
    ytr_wfwht(v, lane);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        dst[n * s_dst + c0 + lane + 32 * e] = ((sm >> e) & 1u) ? -v[e] * YTR_HS : v[e] * YTR_HS;
    }
}

// ---- shortlist head (build_shead.py): the k best rows of a head W for a few inputs h without reading all of W --------------------
// z = proj h (rank 512), int8; score[v] = ascale[v] (sketch8[v] . z8) for all rows; the best rows are scored exactly from an
// int8 copy of W (YSH_M slots: the YSH_MK best and the rows that tie with the last); the k best of those are returned with their logits, in ascending row order.

static constexpr int YSH_R = 512, YSH_M = 1024, YSH_MK = 960, YSH_ZROW = 528, YSH_E = YSH_M / 32;   // M slots: the YSH_MK best rows and their ties

// z[t][j] = proj[j] . x[t]; grid R, 8 warps
static __global__ void ysh_proj_kernel(const float * proj, const float * x, const int64_t sx, const int H, const int n_tok, float * z) {
    __shared__ float part[8];
    const int j = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float * pr = proj + (size_t) j * H;
    float w[24];
#pragma unroll
    for (int l = 0; l < 24; ++l) {
        const int k = threadIdx.x + 256 * l;
        w[l] = k < H ? __ldg(pr + k) : 0.0f;
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    for (int t = 0; t < n_tok; ++t) {
        const float * xt = x + (size_t) t * sx;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 6; ++c) {
            float v[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int k = threadIdx.x + 256 * (4 * c + l);
                v[l] = k < H ? xt[k] : 0.0f;
            }
            acc += (w[4 * c] * v[0] + w[4 * c + 1] * v[1]) + (w[4 * c + 2] * v[2] + w[4 * c + 3] * v[3]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        }
        if (lane == 0) {
            part[warp] = acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                sum += part[k];
            }
            z[(size_t) t * YSH_R + j] = sum;
        }
        __syncthreads();
    }
}

// z8[t] = z[t] as int8 (the scale of a row does not change its ranking); grid n_tok, one warp
static __global__ void ysh_quant_kernel(const float * z, int8_t * z8) {
    const int t = blockIdx.x, lane = threadIdx.x;
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    float v[16], m = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        v[e] = z[(size_t) t * YSH_R + lane + 32 * e];
        m = fmaxf(m, fabsf(v[e]));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    }
    const float a = m > 0.0f ? 127.0f / m : 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        z8[(size_t) t * YSH_R + lane + 32 * e] = (int8_t) max(-127, min(127, __float2int_rn(v[e] * a)));
    }
}

// scores[t][v] for all rows v and up to 16 inputs: grid V / 128, 8 warps of 16 rows; sketch in MMA fragment order
static __global__ void __launch_bounds__(256) ysh_score_kernel(const uint4 * sketch, const float * ascale, const int8_t * z8, const int n_tok,
                                                              const int V, float * scores) {
    __shared__ __align__(16) int8_t s_z[16 * YSH_ZROW];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, tig = lane % 4;
    const int tile = blockIdx.x * 8 + warp;
    const uint4 * sp = sketch + (size_t) tile * 16 * 32 + lane;
    uint4 nx[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        nx[u] = __ldg(sp + 32 * u);
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    for (int k = threadIdx.x; k < 16 * 32; k += 256) {
        const int t = k / 32, c = k % 32;
        *(uint4 *) (s_z + t * YSH_ZROW + 16 * c) = t < n_tok ? *((const uint4 *) (z8 + (size_t) t * YSH_R) + c) : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    int d[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint4 cur[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            cur[u] = nx[u];
        }
        if (c < 3) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                nx[u] = __ldg(sp + 32 * (4 * (c + 1) + u));
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint32_t a[4] = { cur[u].x, cur[u].y, cur[u].z, cur[u].w };
            const int st = 4 * c + u;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int8_t * zr = s_z + (nt * 8 + g) * YSH_ZROW + 32 * st + 4 * tig;
                ytr_mma8(d[nt], a, *(const uint32_t *) zr, *(const uint32_t *) (zr + 16));
            }
        }
    }
    const int   ra = tile * 16 + g, rb = ra + 8;
    const float sa = ascale[ra], sb = ascale[rb];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int t0 = nt * 8 + 2 * tig;
        if (t0 < n_tok)     { scores[(size_t) t0 * V + ra]       = sa * (float) d[nt][0]; scores[(size_t) t0 * V + rb]       = sb * (float) d[nt][2]; }
        if (t0 + 1 < n_tok) { scores[(size_t) (t0 + 1) * V + ra] = sa * (float) d[nt][1]; scores[(size_t) (t0 + 1) * V + rb] = sb * (float) d[nt][3]; }
    }
}

// ex[t][c] = wscale[v] (w8[v] . x[t]), v = ids[t][c]; grid (M / 8, n_tok), one candidate per warp. x[t] is staged with
// element 512 j + 16 l + b at 512 j + 32 b + l, so lane l reads its 16-byte chunks of the row against distinct banks.
static __global__ void __launch_bounds__(256) ysh_rescore_kernel(const int8_t * w8, const float * wscale, const int * ids, const float * x,
                                                                const int64_t sx, const int H, float * ex) {
    extern __shared__ float s_h[];
    const int t = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, c = blockIdx.x * 8 + warp;
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    for (int k = threadIdx.x; k < H; k += 256) {
        s_h[(k / 512) * 512 + 32 * (k % 16) + (k % 512) / 16] = x[(size_t) t * sx + k];
    }
    const int      v   = ids[(size_t) t * YSH_M + c];   // -1: empty slot
    const uint4 *  row = (const uint4 *) (w8 + (size_t) max(v, 0) * H) + lane;
    float acc = 0.0f;
    __syncthreads();
    for (int j0 = 0; j0 < H / 512; j0 += 2) {
        uint4 q[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            q[i] = j0 + i < H / 512 ? row[32 * (j0 + i)] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t wv[4] = { q[i].x, q[i].y, q[i].z, q[i].w };
            const float *  hp    = s_h + min(j0 + i, H / 512 - 1) * 512 + lane;
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                acc += (float) (int8_t) (wv[b / 4] >> (8 * (b % 4))) * hp[32 * b];
            }
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    }
    if (lane == 0) {
        ex[(size_t) t * YSH_M + c] = v >= 0 ? acc * wscale[v] : -INFINITY;
    }
}

// the k (<= 32) best of the M exact scores of an input, in ascending row order: cand[t][k], unary[t][k]; grid n_tok, one warp
static __global__ void ysh_final_kernel(const float * ex, const int * ids, const int k, int * cand, float * unary, int * amax) {
    const int t = blockIdx.x, lane = threadIdx.x;
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    float v[YSH_E];
    int   idv[YSH_E];
#pragma unroll
    for (int e = 0; e < YSH_E; ++e) {
        v[e]   = ex[(size_t) t * YSH_M + lane + 32 * e];
        idv[e] = ids[(size_t) t * YSH_M + lane + 32 * e];
    }
    float sel_v = 0.0f;
    int   sel_id = 0x7FFFFFFF;
    for (int r = 0; r < k; ++r) {
        float m = -INFINITY;
#pragma unroll
        for (int e = 0; e < YSH_E; ++e) {
            m = fmaxf(m, v[e]);
        }
        float wm = m;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            wm = fmaxf(wm, __shfl_xor_sync(0xFFFFFFFFu, wm, o));
        }
        int id = 0x7FFFFFFF;   // of equal scores the lowest row
#pragma unroll
        for (int e = 0; e < YSH_E; ++e) {
            if (v[e] == wm) {
                id = min(id, idv[e]);
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            id = min(id, __shfl_xor_sync(0xFFFFFFFFu, id, o));
        }
#pragma unroll
        for (int e = 0; e < YSH_E; ++e) {
            if (idv[e] == id) {
                v[e] = -INFINITY;
            }
        }
        if (lane == r) {
            sel_v = wm;
            sel_id = id;
        }
        if (r == 0 && lane == 0) {
            amax[t] = id;   // the best row
        }
    }
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        rank += __shfl_sync(0xFFFFFFFFu, sel_id, j) < sel_id ? 1 : 0;
    }
    if (lane < k) {
        cand[(size_t) t * k + rank]  = sel_id;
        unary[(size_t) t * k + rank] = sel_v;
    }
}

// ---- host side ---------------------------------------------------------------------------------------------------------------

struct ytr_bufs {
    float *  acc = nullptr;     // partial sums, YTR_ACC floats, zero between launches
    float *  side = nullptr;    // results of side products [side][n][256] until their nodes are reached
    float *  acc_def = nullptr; // partial sums of deferred products (bump allocated, reset when none is pending)
    int64_t  def_off = 0;
    int *    cnt = nullptr;     // FP16 path: tiles done per 512-row group
    half *   xr  = nullptr;     // FP16 path: rotated input
    int8_t * x8  = nullptr;     // int8 path: rotated input [n][cols]
    float *  al  = nullptr;     // its scales [n][cols / 512]
    const ggml_tensor * x_src = nullptr;   // the tensor x8 (or xr) holds, with x_key and x_epoch
    uint64_t x_key = 0, x_epoch = 0;
};
static std::unordered_map<int, ytr_bufs> g_ytr_bufs;

static ytr_bufs * ytr_get_bufs(const int device, uint64_t & epoch) {
    ytr_bufs * B;
    {
        std::lock_guard<std::mutex> lock(g_ytr_mutex);
        epoch = g_ytr_epoch;
        B = &g_ytr_bufs[device];
    }
    if (B->acc == nullptr) {
        CUDA_CHECK(cudaMalloc(&B->acc, (size_t) YTR_ACC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B->acc_def, (size_t) YTR_ACC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B->side, (size_t) GGML_CUDA_YTR_MAX_SIDE * YTR_NMAX * 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(B->acc_def, 0, (size_t) YTR_ACC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B->cnt, (YTR_RMAX / 512) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&B->xr, (size_t) YTR_NMAX * YTR_CMAX * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&B->x8, (size_t) YTR_NMAX * YTR_CMAX));
        CUDA_CHECK(cudaMalloc(&B->al, (size_t) YTR_NMAX * (YTR_CMAX / 512) * sizeof(float)));
        CUDA_CHECK(cudaMemset(B->acc, 0, (size_t) YTR_ACC * sizeof(float)));
        CUDA_CHECK(cudaMemset(B->cnt, 0, (YTR_RMAX / 512) * sizeof(int)));
    }
    return B;
}

static bool ytr_int8_mode() {
    static const bool off = getenv("GGML_YTR_F16") != nullptr || getenv("GGML_YTR_REF") != nullptr;
    return !off;
}

// a product the int8 kernels take (up to 64 input rows)
static bool ytr_fast(const ggml_tensor * mm, const ytr_info & inf) {
    const ggml_tensor * x = mm->src[1];
    return ytr_int8_mode() && (inf.h.flags & 16) && x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 && x->ne[1] >= 1 &&
        x->ne[1] <= YTR_NMAX && x->ne[2] == 1 && x->ne[3] == 1 && mm->ne[2] == 1 && mm->ne[3] == 1 && x->nb[0] == sizeof(float) &&
        mm->nb[0] == sizeof(float);
}

// L2 prefetch of the start of a product's codes, issued by a plain finish whose next product is not the next kernel (GDN layers:
// out_proj after qkv/z while the recurrence runs). GGML_YTR_L2PF = MiB, default off: prefetching right before a product that
// follows at once made decoding slower (it competes with that product's loads).
static void ytr_pf_of(const ggml_tensor * mm, const char *& p, uint32_t & bytes) {
    static const double mib = getenv("GGML_YTR_L2PF") ? atof(getenv("GGML_YTR_L2PF")) : 0.0;
    p = nullptr;
    bytes = 0;
    if (mib <= 0.0 || mm == nullptr) {
        return;
    }
    const ytr_info inf = ytr_get_info(mm->src[0]);
    const uint64_t total = (uint64_t) inf.h.rows * inf.h.cols / 256 * 16 * inf.h.bits;
    p = (const char *) mm->src[0]->data + inf.h.code_off;
    bytes = (uint32_t) std::min<uint64_t>(total, (uint64_t) (mib * 1048576.0)) & ~15u;
}

// side products computed by a launch, waiting for their node (see ggml_cuda_ytr_side_finish)
struct ytr_side_res { float * buf; int rows, n_tok; };
static std::unordered_map<const ggml_tensor *, ytr_side_res> g_ytr_side;
static void ytr_side_clear() { g_ytr_side.clear(); }

struct ytr_job {
    const ggml_tensor * node;   // the MUL_MAT
    ytr_info            inf;
    float *             acc;    // its partial sums [n][rows]; null: the result is written directly
    bool                deferred = false;   // partial sums kept in acc_def, finished when the graph reaches the node
};

// products computed by an earlier launch whose result memory was still in use there: finished at their own node
// (g_ytr_defer_n mirrors the keys for the check in ggml_cuda_ytr_prepare)
static std::unordered_map<const ggml_tensor *, ytr_job> g_ytr_defer;


// finish a deferred product at its node; false if node is not deferred
static bool ytr_finish_deferred(ytr_bufs * B, const ggml_tensor * node, cudaStream_t stream);


// split-K workers of a launch with `tiles` 16-row tiles in all over G groups of 512 columns
static int ytr_workers(const int G, const int tiles, const int nsm) {
    static const int want = getenv("GGML_YTR_WANT") ? atoi(getenv("GGML_YTR_WANT")) : 32;   // warps per SM to aim for
    const int nw  = std::max(1, std::min(G, (nsm * want + tiles - 1) / tiles));
    const int per = (G + nw - 1) / nw;
    return (G + per - 1) / per;   // fewest workers for the same groups per worker
}

static void ytr_ensure_x8(ytr_bufs * B, const ggml_tensor * x, const ggml_tensor * mm, const ytr_info & inf, const uint64_t epoch,
                          cudaStream_t stream) {
    if (B->x_src == x && B->x_key == inf.in_key && B->x_epoch == epoch) {
        return;
    }
    const uint8_t * base = (const uint8_t *) mm->src[0]->data;
    const int cols = inf.h.cols, G = cols / 512, n_tok = (int) x->ne[1];
    const char * pf = nullptr;
    const uint32_t pfb = 0;
    ggml_cuda_kernel_launch(ytr_in8_kernel, ggml_cuda_kernel_launch_params(dim3((G + 3) / 4, n_tok), dim3(128), 0, stream),
        (const float *) x->data, (int64_t) (x->nb[1] / sizeof(float)), base + inf.h.sin_off,
        (const int32_t *) ((inf.h.flags & 4) ? base + inf.h.cmap_off : nullptr), cols, B->x8, B->al, pf, pfb);
    B->x_src = x; B->x_key = inf.in_key; B->x_epoch = epoch;
}

template <int K, int NT, int UW, int MINB>
static void ytr_gv8_go(const ytr_gv_args & A, const int ctas, const int nw, const size_t smem, cudaStream_t stream) {
    ggml_cuda_kernel_launch(ytr_gv8_kernel<K, NT, UW, MINB>, ggml_cuda_kernel_launch_params(dim3(ctas, nw), dim3(UW * 32), smem, stream), A);
}

// the int8 products of jobs (same input in x8, same bits); sets jobs[s].acc
static void ytr_gemv(ytr_bufs * B, ytr_job * jobs, const int n_jobs, const int n_tok, const int nsm, cudaStream_t stream,
                     const ggml_tensor * x = nullptr, const ggml_cuda_ytr_side * sides = nullptr, int n_sides = 0, bool * sides_taken = nullptr) {
    ytr_gv_args A = {};
    A.x8 = B->x8; A.alpha = B->al; A.n_seg = n_jobs; A.n_tok = n_tok; A.cols = jobs[0].inf.h.cols;
    const int K  = jobs[0].inf.h.bits;
    static const bool gw_on = !(getenv("GGML_YTR_GW") && atoi(getenv("GGML_YTR_GW")) == 0);
    const int UW = (n_tok <= 16 || gw_on) ? 8 : 4;
    int64_t acc_off = 0;
    int ctas = 0, max_nw = 1, tiles = 0;
    for (int s = 0; s < n_jobs; ++s) {
        tiles += (jobs[s].inf.h.flags & 3) ? jobs[s].inf.h.rows / 16 : 0;
    }
    const int nw_all = ytr_workers(A.cols / 512, std::max(tiles, 1), nsm);
    for (int s = 0; s < n_jobs; ++s) {
        const ytr_header & h = jobs[s].inf.h;
        GGML_ASSERT((int) h.bits == K && (int) h.cols == A.cols);
        const uint8_t * base = (const uint8_t *) jobs[s].node->src[0]->data;
        ytr_seg & S = A.seg[s];
        S.codes = (const uint4 *) (base + h.code_off);
        S.scale = (const half *) (base + h.scale_off);
        S.lut8  = jobs[s].inf.lut8;
        S.gamma = jobs[s].inf.gamma;
        S.rows  = h.rows;
        S.direct = !(h.flags & 3);
        if (S.direct) {
            S.nw  = 1;
            S.out = (float *) jobs[s].node->data;
            S.so  = jobs[s].node->nb[1] / sizeof(float);
            jobs[s].acc = nullptr;
        } else if (jobs[s].deferred) {
            S.nw  = nw_all;
            S.out = B->acc_def + B->def_off;
            S.so  = h.rows;
            jobs[s].acc = S.out;
            B->def_off += (int64_t) n_tok * h.rows;
            GGML_ASSERT(B->def_off <= YTR_ACC);
        } else {
            S.nw  = nw_all;
            S.out = B->acc + acc_off;
            S.so  = h.rows;
            jobs[s].acc = S.out;
            acc_off += (int64_t) n_tok * h.rows;
            GGML_ASSERT(acc_off <= YTR_ACC);
        }
        S.cta_begin = ctas;
        ctas += (h.rows / 16 + UW - 1) / UW;
        max_nw = std::max(max_nw, S.nw);
    }
    // side rows as extra blocks (up to 16 input rows: the kernels with 8-warp blocks)
    if (n_sides > 0 && x != nullptr && n_tok <= 16 && n_sides <= GGML_CUDA_YTR_MAX_SIDE) {
        bool ok = true;
        int rows = 0;
        for (int s = 0; s < n_sides; ++s) {
            const ggml_tensor * w = sides[s].w;
            const int wt = w->type == GGML_TYPE_F32 ? 0 : w->type == GGML_TYPE_F16 ? 1 : w->type == GGML_TYPE_BF16 ? 2 : -1;
            ok = ok && wt >= 0 && w->ne[0] == A.cols && w->ne[2] == 1 && w->ne[3] == 1;
            rows += (int) w->ne[1];
            ok = ok && w->ne[1] <= 256;
            A.side_w[s] = (const char *) w->data; A.side_wrow[s] = w->nb[1]; A.side_type[s] = wt; A.side_end[s] = rows;
            A.side_out[s] = B->side + (size_t) s * YTR_NMAX * 256; A.side_col[s] = 256; A.side_post[s] = sides[s].post_op;
            A.side_pre[s] = sides[s].pre; A.side_mul[s] = sides[s].mul;
        }
        if (ok) {
            A.n_side = n_sides; A.side_begin = ctas; A.xf = (const float *) x->data; A.sx = x->nb[1] / sizeof(float);
            ctas += rows;
            std::lock_guard<std::mutex> lock(g_ytr_mutex);
            for (int s = 0; s < n_sides; ++s) {
                g_ytr_side[sides[s].node] = { A.side_out[s], (int) sides[s].w->ne[1], n_tok };
            }
            if (sides_taken) {
                *sides_taken = true;
            }
        }
    }
    // inputs staged per block: up to its worker's groups, as many as fit in 24 KiB (36 KiB above 32 rows: one group of 64)
    const int    G    = A.cols / 512;
    const size_t per  = (size_t) n_tok * (YTR_XROW + sizeof(float));
    const int    fit  = (int) ((n_tok > 32 ? 36864 : 24576) / per);
    A.sg = std::max(1, std::min(fit, (G + max_nw - 1) / max_nw));
    const size_t smem = (size_t) A.sg * per;
    static const bool gw = !(getenv("GGML_YTR_GW") && atoi(getenv("GGML_YTR_GW")) == 0);   // 17..64 rows: ytr_gw8_kernel
    const size_t smem_gw = (size_t) 8 * 16 * YTR_GW_AROW + (size_t) (n_tok <= 32 ? 32 : 64) * (YTR_GW_BROW + sizeof(float));
#define YTR_GV(KK)                                                                       \
    if      (n_tok <= 8)  ytr_gv8_go<KK, 1, 8, 4>(A, ctas, max_nw, smem, stream);                 \
    else if (n_tok <= 16) ytr_gv8_go<KK, 2, 8, 3>(A, ctas, max_nw, smem, stream);                 \
    else if (gw && n_tok <= 32) ggml_cuda_kernel_launch(ytr_gw8_kernel<KK, 4>, ggml_cuda_kernel_launch_params(dim3(ctas, max_nw), dim3(256), smem_gw, stream), A); \
    else if (gw)          ggml_cuda_kernel_launch(ytr_gw8_kernel<KK, 8>, ggml_cuda_kernel_launch_params(dim3(ctas, max_nw), dim3(256), smem_gw, stream), A); \
    else if (n_tok <= 32) ytr_gv8_go<KK, 4, 4, 3>(A, ctas, max_nw, smem, stream);                 \
    else                  ytr_gv8_go<KK, 8, 4, 2>(A, ctas, max_nw, smem, stream);
    if (K == 6) {
        YTR_GV(6)
    } else {
        YTR_GV(8)
    }
#undef YTR_GV
}

// plain finish of the jobs with partial sums (all but `skip_mask`)
static void ytr_fin(const ytr_job * jobs, const int n_jobs, const unsigned skip_mask, const int n_tok, const ggml_tensor * pf_node,
                    cudaStream_t stream) {
    ytr_fin_args F = {};
    int grp = 0;
    for (int s = 0; s < n_jobs; ++s) {
        if (jobs[s].acc == nullptr || jobs[s].deferred || (skip_mask >> s) & 1) {
            continue;
        }
        const ytr_header & h = jobs[s].inf.h;
        const uint8_t * base = (const uint8_t *) jobs[s].node->src[0]->data;
        ytr_fin_seg & S = F.seg[F.n_seg++];
        S.sout = (h.flags & 1) ? base + h.sout_off : nullptr;
        S.rinv = jobs[s].inf.rinv;
        S.acc  = jobs[s].acc;
        S.dst  = (float *) jobs[s].node->data;
        S.sy   = jobs[s].node->nb[1] / sizeof(float);
        S.rows = h.rows;
        S.grp_begin = grp;
        grp += h.rows / 512;
    }
    if (F.n_seg == 0) {
        return;
    }
    F.n_tok = n_tok;
    ytr_pf_of(pf_node, F.pf, F.pf_bytes);
    ggml_cuda_kernel_launch(ytr_fin_kernel, ggml_cuda_kernel_launch_params(dim3(grp, (n_tok + 7) / 8), dim3(256), 0, stream), F);
}

// remove a deferred product from the records; false if node is not deferred
static bool ytr_take_deferred(ytr_bufs * B, const ggml_tensor * node, ytr_job & job) {
    std::lock_guard<std::mutex> lock(g_ytr_mutex);
    auto it = g_ytr_defer.find(node);
    if (it == g_ytr_defer.end()) {
        return false;
    }
    job = it->second;
    job.deferred = false;
    g_ytr_defer.erase(it);
    g_ytr_defer_n.erase(node);
    if (g_ytr_defer.empty()) {
        B->def_off = 0;   // stream-ordered: later launches reuse the space after this finish
    }
    return true;
}

static bool ytr_finish_deferred(ytr_bufs * B, const ggml_tensor * node, cudaStream_t stream) {
    ytr_job job;
    if (!ytr_take_deferred(B, node, job)) {
        return false;
    }
    ytr_fin(&job, 1, 0u, (int) node->src[1]->ne[1], nullptr, stream);
    return true;
}

// more than 64 input rows: int8 input, weights decoded to int8 (row chunks of at most 128 MiB for weights without output rotation,
// such as the output head), int8 GEMM, then the finish
static void ytr_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ytr_info & inf) {
    const ytr_header & h = inf.h;
    const uint8_t * base = (const uint8_t *) src0->data;
    const int N = (int) src1->ne[1], cols = h.cols, rows = h.rows, G = cols / 512;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> x8(ctx.pool(), (size_t) N * cols);
    ggml_cuda_pool_alloc<float>  al(ctx.pool(), (size_t) N * G);
    ggml_cuda_kernel_launch(ytr_in8_kernel, ggml_cuda_kernel_launch_params(dim3((G + 3) / 4, N), dim3(128), 0, stream),
        (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), base + h.sin_off,
        (const int32_t *) ((h.flags & 4) ? base + h.cmap_off : nullptr), cols, x8.get(), al.get(), (const char *) nullptr, 0u);
    const bool direct = !(h.flags & 3);
    const int  cr     = direct ? std::min(rows, std::max(128, (int) (((size_t) 128 << 20) / cols) / 128 * 128)) : rows;
    ggml_cuda_pool_alloc<int8_t> w8(ctx.pool(), (size_t) cr * cols);
    ggml_cuda_pool_alloc<float>  acc(ctx.pool(), direct ? 1 : (size_t) N * rows);
    const int64_t sy = dst->nb[1] / sizeof(float);
    for (int r0 = 0; r0 < rows; r0 += cr) {
        const int nr = std::min(cr, rows - r0);
        const ggml_cuda_kernel_launch_params pd(dim3(cols / 256, nr / 128), dim3(256), 0, stream);
        if (h.bits == 6) {
            ggml_cuda_kernel_launch(ytr_predecode_kernel<6>, pd, (const uint4 *) (base + h.code_off), (const uint16_t *) inf.lut8, cols, r0, w8.get());
        } else {
            ggml_cuda_kernel_launch(ytr_predecode_kernel<8>, pd, (const uint4 *) (base + h.code_off), (const uint16_t *) inf.lut8, cols, r0, w8.get());
        }
        float * out = direct ? (float *) dst->data + r0 : acc.get() + r0;
        ggml_cuda_kernel_launch(ytr_gemm8_kernel, ggml_cuda_kernel_launch_params(dim3(nr / 128, (N + 63) / 64), dim3(256), 0, stream),
            (const int8_t *) w8.get(), (const int8_t *) x8.get(), (const float *) al.get(), (const half *) (base + h.scale_off) + r0, inf.gamma,
            out, direct ? sy : (int64_t) rows, N, cols);
    }
    if (!direct) {
        ytr_fin_args F = {};
        F.n_seg = 1; F.n_tok = N; F.keep = 1;
        F.seg[0] = { (h.flags & 1) ? base + h.sout_off : nullptr, inf.rinv, acc.get(), (float *) dst->data, sy, rows, 0 };
        ggml_cuda_kernel_launch(ytr_fin_kernel, ggml_cuda_kernel_launch_params(dim3(rows / 512, (N + 7) / 8), dim3(256), 0, stream), F);
    }
}

// ---- FP16 path (GGML_YTR_F16=1) ------------------------------------------------------------------------------------------------

template <int K, int NT>
static void ytr_launch(const ytr_header & h, const uint8_t * base, const half * xr, int n_tok, float * acc, int * cnt, const int * rinv,
                       float * dst, int64_t sy, int nsm, cudaStream_t stream) {
    // tuning: GGML_YTR_KC = most 512-column blocks per tile, GGML_YTR_GRID = blocks per SM
    static const int kc_max   = getenv("GGML_YTR_KC")   ? atoi(getenv("GGML_YTR_KC"))   : 2;
    static const int grid_per = getenv("GGML_YTR_GRID") ? atoi(getenv("GGML_YTR_GRID")) : 3;
    const int n_rt = h.rows / 128, n_kb = h.cols / 512;
    int kc = n_kb;
    if (h.flags & 3) {
        kc = kc_max;
        while (kc > 1 && n_rt * ((n_kb + kc - 1) / kc) < 2 * nsm) {
            kc /= 2;
        }
    }
    const int n_tiles = n_rt * ((n_kb + kc - 1) / kc);
    const int grid    = std::min(n_tiles, nsm * grid_per);
    ytr_mm_kernel<K, NT><<<grid, 256, 0, stream>>>(base, h, xr, n_tok, kc, n_tiles, acc, cnt, rinv, dst, sy, (h.flags & 3) != 0);
    CUDA_CHECK(cudaGetLastError());
    if (h.flags & 3) {
        ytr_fin_args F = {};
        F.n_seg = 1; F.n_tok = n_tok;
        F.seg[0] = { (h.flags & 1) ? base + h.sout_off : nullptr, rinv, acc, dst, sy, (int) h.rows, 0 };
        ytr_fin_kernel<<<dim3(h.rows / 512, (n_tok + 7) / 8), 256, 0, stream>>>(F);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_get_rows_ytr(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const ytr_info inf = ytr_get_info(src0);
    const ytr_header & h = inf.h;
    GGML_ASSERT((h.flags & 31) == 0 && "YTR GET_ROWS: row-major codes without rotation of the rows or maps only");
    GGML_ASSERT(src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && src1->ne[1] == 1 && src1->ne[2] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1);
    const int n = (int) src1->ne[0], G = h.cols / 512;
    if (n == 0) {
        return true;
    }
    ggml_cuda_kernel_launch(ytr_rows_kernel, ggml_cuda_kernel_launch_params(dim3((G + 3) / 4, n), dim3(128), 0, ctx.stream()),
        (const uint8_t *) src0->data, h, (const int32_t *) src1->data, (int64_t) (src1->nb[0] / sizeof(int32_t)), (float *) dst->data,
        (int64_t) (dst->nb[1] / sizeof(float)));
    return true;
}

bool ggml_cuda_ytr_side_finish(ggml_backend_cuda_context & ctx, const ggml_tensor * node, float * out, const int64_t out_col) {
    ytr_side_res r;
    {
        std::lock_guard<std::mutex> lock(g_ytr_mutex);
        auto it = g_ytr_side.find(node);
        if (it == g_ytr_side.end()) {
            return false;
        }
        r = it->second;
        g_ytr_side.erase(it);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(out, out_col * sizeof(float), r.buf, 256 * sizeof(float), r.rows * sizeof(float), r.n_tok,
                                 cudaMemcpyDeviceToDevice, ctx.stream()));
    return true;
}

// ---- shortlist head: host side -------------------------------------------------------------------------------------------------

struct ytr_shead {
    bool tried = false, ok = false;
    int V = 0, H = 0;
    float * proj = nullptr, * ascale = nullptr, * wscale = nullptr;
    uint4 * sketch = nullptr;
    int8_t * w8 = nullptr;
    // work buffers for up to 16 inputs
    float * z = nullptr, * scores = nullptr, * ex = nullptr, * unary = nullptr;
    int8_t * z8 = nullptr;
    int * ids = nullptr, * cand = nullptr, * amax = nullptr;
};
static ytr_shead g_ysh;
struct ytr_shead_out { const void * buf; size_t bytes; };
static std::unordered_map<const ggml_tensor *, ytr_shead_out> g_ysh_pending;   // TOP_K and GET_ROWS results, copied out at their nodes
static void ytr_shead_clear() { g_ysh_pending.clear(); }

// GGML_YTR_SHEAD=<file from build_shead.py>: loaded at the first use by a head of its size
static bool ytr_shead_get(const int V, const int H) {
    if (!g_ysh.tried) {
        g_ysh.tried = true;
        const char * path = getenv("GGML_YTR_SHEAD");
        FILE * f = path ? fopen(path, "rb") : nullptr;
        if (f) {
            struct { char magic[4]; uint32_t V, H, R; uint64_t off[5]; } hd;
            if (fread(&hd, sizeof(hd), 1, f) == 1 && memcmp(hd.magic, "YSH1", 4) == 0 && hd.R == YSH_R &&
                    hd.V == (uint32_t) V && hd.H == (uint32_t) H && hd.V >= YSH_MK && hd.V % 128 == 0 && hd.H > 0 && hd.H % 512 == 0 && hd.H <= 24 * 256) {
                const size_t sz[5] = { (size_t) hd.R * hd.H * 4, (size_t) hd.V * 4, (size_t) hd.V * hd.R, (size_t) hd.V * 4, (size_t) hd.V * hd.H };
                void * dev[5] = {};
                std::vector<char> buf;
                bool ok = fseek(f, 0, SEEK_END) == 0;
                const long file_size = ok ? ftell(f) : -1;
                ok = file_size >= 0;
                for (int i = 0; ok && i < 5; ++i) {
                    ok = hd.off[i] >= 64 && hd.off[i] <= (uint64_t) file_size && sz[i] <= (uint64_t) file_size - hd.off[i];
                }
                for (int i = 0; ok && i < 5; ++i) {
                    buf.resize(sz[i]);
                    ok = fseek(f, (long) hd.off[i], SEEK_SET) == 0 && fread(buf.data(), 1, sz[i], f) == sz[i];
                    if (ok) {
                        CUDA_CHECK(cudaMalloc(&dev[i], sz[i]));
                        CUDA_CHECK(cudaMemcpy(dev[i], buf.data(), sz[i], cudaMemcpyHostToDevice));
                    }
                }
                if (ok) {
                    g_ysh.V = hd.V; g_ysh.H = hd.H;
                    g_ysh.proj = (float *) dev[0]; g_ysh.ascale = (float *) dev[1]; g_ysh.sketch = (uint4 *) dev[2];
                    g_ysh.wscale = (float *) dev[3]; g_ysh.w8 = (int8_t *) dev[4];
                    CUDA_CHECK(cudaMalloc(&g_ysh.z, 16 * YSH_R * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.z8, 16 * YSH_R));
                    CUDA_CHECK(cudaMalloc(&g_ysh.scores, (size_t) 16 * hd.V * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.ids, 16 * YSH_M * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.ex, 16 * YSH_M * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.cand, 16 * 32 * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.amax, 16 * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&g_ysh.unary, 16 * 32 * sizeof(float)));
                    g_ysh.ok = true;
                    GGML_LOG_INFO("%s: shortlist head %s: %u rows, rank %u\n", __func__, path, hd.V, hd.R);
                } else {
                    for (void * p : dev) {
                        if (p != nullptr) {
                            CUDA_CHECK(cudaFree(p));
                        }
                    }
                }
            }
            fclose(f);
            if (!g_ysh.ok) {
                GGML_LOG_WARN("%s: could not use the shortlist head %s\n", __func__, path);
            }
        }
    }
    return g_ysh.ok && g_ysh.V == V && g_ysh.H == H;
}

// node i = head product (no output rotation) whose result only feeds TOP_K (k <= 32), GET_ROWS of that top-k (the DFlash2 selector)
// and possibly an ARGMAX: compute those from the shortlist head instead; the full logits are not made (they may be flagged as a
// graph output: a DFlash2 draft pass has no output rows to read). Returns false if the pattern or the head does not fit.
static bool ytr_shead_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i, const ytr_info & inf) {
    ggml_tensor * node = cgraph->nodes[i];
    const ggml_tensor * x = node->src[1];
    const int n_tok = (int) x->ne[1];
    if ((inf.h.flags & 3) || n_tok > 16) {
        return false;
    }
    const ggml_tensor * topk = nullptr, * resh = nullptr, * rows = nullptr, * amax = nullptr;
    int n_use = 0;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * m = cgraph->nodes[j];
        bool reads = false;
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            reads = reads || m->src[q] == node || (m->src[q] != nullptr && m->src[q]->view_src == node && m->src[q] != resh);
        }
        if (m->op == GGML_OP_TOP_K && m->src[0] == node && topk == nullptr) {
            topk = m; n_use++;
        } else if (m->op == GGML_OP_RESHAPE && m->src[0] == node && resh == nullptr && ggml_node_get_use_count(cgraph, j) == 1 &&
                   !(m->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            resh = m; n_use++;
        } else if (m->op == GGML_OP_ARGMAX && m->src[0] == node && amax == nullptr && m->type == GGML_TYPE_I32 && ggml_nelements(m) == n_tok) {
            amax = m; n_use++;
        } else if (m->op == GGML_OP_GET_ROWS && resh != nullptr && m->src[0] == resh && m->src[1] == topk && rows == nullptr) {
            rows = m;
        } else if (reads) {
            return false;   // another reader of the logits
        }
    }
    if (topk == nullptr || rows == nullptr || strcmp(topk->name, "dflash2_candidates") != 0 || ggml_node_get_use_count(cgraph, i) != n_use) {
        return false;
    }
    const int k = (int) topk->ne[0];
    if (k < 1 || k > 32 || topk->ne[1] != n_tok || topk->type != GGML_TYPE_I32 || rows->type != GGML_TYPE_F32 || !ggml_is_contiguous(topk) ||
            !ggml_is_contiguous(rows) || ggml_nelements(rows) != (int64_t) k * n_tok || resh->ne[0] != 1 || resh->ne[1] != inf.h.rows) {
        return false;
    }
    if (!ytr_shead_get((int) inf.h.rows, (int) inf.h.cols)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t sx = x->nb[1] / sizeof(float);
    const int V = g_ysh.V, H = g_ysh.H;
    ggml_cuda_kernel_launch(ysh_proj_kernel, ggml_cuda_kernel_launch_params(dim3(YSH_R), dim3(256), 0, stream),
        (const float *) g_ysh.proj, (const float *) x->data, sx, H, n_tok, g_ysh.z);
    ggml_cuda_kernel_launch(ysh_quant_kernel, ggml_cuda_kernel_launch_params(dim3(n_tok), dim3(32), 0, stream), (const float *) g_ysh.z, g_ysh.z8);
    ggml_cuda_kernel_launch(ysh_score_kernel, ggml_cuda_kernel_launch_params(dim3(V / 128), dim3(256), 0, stream),
        (const uint4 *) g_ysh.sketch, (const float *) g_ysh.ascale, (const int8_t *) g_ysh.z8, n_tok, V, g_ysh.scores);
    if (!ggml_cuda_top_k_rows(ctx, g_ysh.scores, g_ysh.ids, V, n_tok, YSH_MK, YSH_M)) {
        return false;
    }
    ggml_cuda_kernel_launch(ysh_rescore_kernel, ggml_cuda_kernel_launch_params(dim3(YSH_M / 8, n_tok), dim3(256), (size_t) H * sizeof(float), stream),
        (const int8_t *) g_ysh.w8, (const float *) g_ysh.wscale, (const int *) g_ysh.ids, (const float *) x->data, sx, H, g_ysh.ex);
    ggml_cuda_kernel_launch(ysh_final_kernel, ggml_cuda_kernel_launch_params(dim3(n_tok), dim3(32), 0, stream),
        (const float *) g_ysh.ex, (const int *) g_ysh.ids, k, g_ysh.cand, g_ysh.unary, g_ysh.amax);
    std::lock_guard<std::mutex> lock(g_ytr_mutex);
    if (amax) {
        g_ysh_pending[amax] = { g_ysh.amax, (size_t) n_tok * sizeof(int) };
    }
    g_ysh_pending[topk] = { g_ysh.cand, (size_t) k * n_tok * sizeof(int) };
    g_ysh_pending[rows] = { g_ysh.unary, (size_t) k * n_tok * sizeof(float) };
    g_ytr_done.insert(node);
    return true;
}

bool ggml_cuda_ytr_node_hook(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    if (node->op != GGML_OP_TOP_K && node->op != GGML_OP_GET_ROWS && node->op != GGML_OP_ARGMAX) {
        return false;
    }
    ytr_shead_out o;
    {
        std::lock_guard<std::mutex> lock(g_ytr_mutex);
        auto it = g_ysh_pending.find(node);
        if (it == g_ysh_pending.end()) {
            return false;
        }
        o = it->second;
        g_ysh_pending.erase(it);
    }
    CUDA_CHECK(cudaMemcpyAsync(node->data, o.buf, o.bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
    return true;
}

// ---- entry points ------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_ytr(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    {
        std::lock_guard<std::mutex> lock(g_ytr_mutex);
        if (g_ytr_done.count(dst)) {
            return;   // computed by ggml_cuda_ytr_try_fuse
        }
    }
    if (g_ytr_defer_n.count(dst)) {
        uint64_t ep;
        ytr_finish_deferred(ytr_get_bufs(ctx.device, ep), dst, ctx.stream());
        return;
    }

    const ytr_info   inf  = ytr_get_info(src0);
    const ytr_header h    = inf.h;
    const uint8_t *  base = (const uint8_t *) src0->data;
    const int64_t n_tok = src1->ne[1], cols = h.cols;
    const int64_t sx = src1->nb[1] / sizeof(float), sy = dst->nb[1] / sizeof(float);
    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    static const bool ref = getenv("GGML_YTR_REF") != nullptr;

    if (ytr_fast(dst, inf)) {
        uint64_t epoch;
        ytr_bufs * B = ytr_get_bufs(ctx.device, epoch);
        ytr_ensure_x8(B, src1, dst, inf, epoch, stream);
        ytr_job job = { dst, inf, nullptr };
        ytr_gemv(B, &job, 1, (int) n_tok, nsm, stream);
        ytr_fin(&job, 1, 0u, (int) n_tok, nullptr, stream);
        return;
    }
    if (ytr_int8_mode() && (h.flags & 16) && n_tok > YTR_NMAX) {
        ytr_prefill(ctx, src0, src1, dst, inf);
        return;
    }

    if (ref || n_tok > YTR_NMAX || !(h.flags & 16)) {
        ggml_cuda_pool_alloc<half> xt(ctx.pool(), n_tok * cols);
        ytr_in_kernel<<<dim3(cols / 512, n_tok), 256, 0, stream>>>((const float *) src1->data, sx, base, h, xt.get(), 0);
        CUDA_CHECK(cudaGetLastError());
        ytr_mul_mat_ref(ctx, h, base, xt.get(), n_tok, (float *) dst->data, sy);
        return;
    }

    // FP16 path
    uint64_t epoch;
    ytr_bufs * B = ytr_get_bufs(ctx.device, epoch);
    if (B->x_src != src1 || B->x_key != (inf.in_key ^ 1) || B->x_epoch != epoch) {
        ytr_in_kernel<<<dim3(cols / 512, n_tok), 256, 0, stream>>>((const float *) src1->data, sx, base, h, B->xr, (int) ((n_tok + 7) / 8 * 8));
        CUDA_CHECK(cudaGetLastError());
        B->x_src = src1; B->x_key = inf.in_key ^ 1; B->x_epoch = epoch;   // ^ 1: not the int8 input
    }
    float * d = (float *) dst->data;
#define YTR_GO(KK, NN) ytr_launch<KK, NN>(h, base, B->xr, (int) n_tok, B->acc, B->cnt, inf.rinv, d, sy, nsm, stream)
    if (h.bits == 6) {
        if      (n_tok <= 8)  YTR_GO(6, 1);
        else if (n_tok <= 16) YTR_GO(6, 2);
        else if (n_tok <= 32) YTR_GO(6, 4);
        else                  YTR_GO(6, 8);
    } else {
        if      (n_tok <= 8)  YTR_GO(8, 1);
        else if (n_tok <= 16) YTR_GO(8, 2);
        else if (n_tok <= 32) YTR_GO(8, 4);
        else                  YTR_GO(8, 8);
    }
#undef YTR_GO
}

static bool ytr_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// [D, n] F32 with contiguous rows
static bool ytr_rows_f32(const ggml_tensor * t, const int64_t D, const int64_t n) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == D && t->ne[1] == n && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
}

// node k is read only by node r (directly or through views among the nodes in between) and is not a graph output: a kernel
// that computes node r itself need not write it
static bool ytr_read_only_by(const ggml_cgraph * cgraph, const int k, const int r) {
    const ggml_tensor * t = cgraph->nodes[k];
    if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, k) != 1) {
        return false;
    }
    for (int j = k + 1; j <= r; ++j) {
        const ggml_tensor * m = cgraph->nodes[j];
        bool reads = false;
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            reads = reads || m->src[q] == t;
        }
        if (!reads) {
            continue;
        }
        if (j == r) {
            return true;
        }
        if (!ytr_noop(m) || m->view_src != t || (m->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, j) != 1) {
            return false;
        }
        t = m;   // follow the view
    }
    return false;
}

// the first YTR product within 64 nodes after node k that reads x (any input if x is null) and the int8 path takes
static const ggml_tensor * ytr_next_product(const ggml_cgraph * cgraph, const int k, const ggml_tensor * x, ytr_info * inf) {
    for (int j = k + 1; j < cgraph->n_nodes && j <= k + 64; ++j) {
        const ggml_tensor * mm = cgraph->nodes[j];
        if (mm->op != GGML_OP_MUL_MAT || !ggml_is_ytr(mm->src[0]->type) || !(mm->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        if (x != nullptr && mm->src[1] != x) {
            continue;
        }
        const ytr_info i2 = ytr_get_info(mm->src[0]);
        if (!ytr_fast(mm, i2)) {
            return nullptr;
        }
        if (inf) {
            *inf = i2;
        }
        return mm;
    }
    return nullptr;
}

int ggml_cuda_ytr_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i, const ggml_cuda_ytr_side * sides, const int n_sides,
                           bool * sides_taken) {
    static const bool no_fuse = getenv("GGML_YTR_FUSE") && atoi(getenv("GGML_YTR_FUSE")) == 0;   // one product per launch
    ggml_tensor * node = cgraph->nodes[i];
    {
        uint64_t ep;
        if (!g_ytr_defer.empty() && ytr_finish_deferred(ytr_get_bufs(ctx.device, ep), node, ctx.stream())) {
            std::lock_guard<std::mutex> lock(g_ytr_mutex);
            g_ytr_done.insert(node);
            return 0;
        }
    }
    const ytr_info inf0 = ytr_get_info(node->src[0]);
    if (!ytr_fast(node, inf0)) {
        return -1;
    }
    const ggml_tensor * x = node->src[1];
    const int n_tok = (int) x->ne[1];
    const int n_nodes = cgraph->n_nodes;
    if (!(inf0.h.flags & 3) && ytr_shead_try(ctx, cgraph, i, inf0)) {
        return 0;
    }

    {   // GGML_YTR_SHEAD_DUMP=<file>: append the drafter's hidden states (inputs of the output head) as raw F32 rows, for
        // build_shead.py. Needs GGML_CUDA_DISABLE_GRAPHS=1 (a host copy cannot be captured).
        static FILE * dump = [] { const char * e = getenv("GGML_YTR_SHEAD_DUMP"); return e ? fopen(e, "ab") : nullptr; }();
        if (dump && !(inf0.h.flags & 3) && strcmp(cgraph->nodes[0]->name, "inp_noise_embd") == 0) {
            std::vector<float> hb((size_t) inf0.h.cols);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            for (int t = 0; t < n_tok; ++t) {
                CUDA_CHECK(cudaMemcpy(hb.data(), (const char *) x->data + t * x->nb[1], hb.size() * sizeof(float), cudaMemcpyDeviceToHost));
                fwrite(hb.data(), sizeof(float), hb.size(), dump);
            }
            fflush(dump);
        }
    }

    // products of the same input after node i, computed now: the input must be unchanged until their place. Their results are
    // written now if nothing between reads or writes that memory, else at their own node (deferred finish)
    static const bool no_defer = getenv("GGML_YTR_DEFER") && atoi(getenv("GGML_YTR_DEFER")) == 0;
    ytr_job jobs[YTR_MAX_SEG];
    int     jidx[YTR_MAX_SEG];
    int     nj = 0;
    jobs[nj] = { node, inf0, nullptr }; jidx[nj++] = i;
    int64_t rows_tot = inf0.h.rows;
    for (int j = i + 1; !no_fuse && (inf0.h.flags & 3) && j < n_nodes && j <= i + 64 && nj < YTR_MAX_SEG; ++j) {
        ggml_tensor * mm = cgraph->nodes[j];
        if (mm->op != GGML_OP_MUL_MAT || !ggml_is_ytr(mm->src[0]->type) || !(mm->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                ctx.precomputed.count(mm)) {
            continue;
        }
        if (!(mm->src[1] == x || (mm->src[1]->data == x->data && ggml_are_same_shape(mm->src[1], x) && ggml_are_same_stride(mm->src[1], x)))) {
            continue;
        }
        const ytr_info inf = ytr_get_info(mm->src[0]);
        if (!ytr_fast(mm, inf) || inf.in_key != inf0.in_key || inf.h.bits != inf0.h.bits || !(inf.h.flags & 3) ||
                (rows_tot + inf.h.rows) * n_tok > YTR_ACC) {
            continue;
        }
        bool in_ok = true, out_ok = true;
        for (int k = i; in_ok && k < j; ++k) {
            const ggml_tensor * m = cgraph->nodes[k];
            if (ytr_noop(m)) {
                continue;
            }
            in_ok  = k == i || !ggml_cuda_ranges_overlap(m, x);
            out_ok = out_ok && !ggml_cuda_ranges_overlap(m, mm);
            for (int q = 0; out_ok && q < GGML_MAX_SRC; ++q) {
                out_ok = !ggml_cuda_ranges_overlap(m->src[q], mm);
            }
        }
        for (int s = 0; out_ok && s < nj; ++s) {
            out_ok = !ggml_cuda_ranges_overlap(jobs[s].node, mm);
        }
        if (!in_ok || (!out_ok && no_defer)) {
            continue;
        }
        jobs[nj] = { mm, inf, nullptr, !out_ok }; jidx[nj++] = j;
        rows_tot += inf.h.rows;
    }

    // epilogue folded into node i's finish: E2 = gate + up + SWIGLU (+ down input), E1 = ADD + RMS_NORM + MUL (+ next input)
    int kind = 0, skip = 0, gj = -1, uj = -1, ka_i = -1, kr_i = -1;
    ggml_tensor * glu = nullptr, * add = nullptr, * rms = nullptr, * mul = nullptr;
    const ggml_tensor * res = nullptr;
    if (!no_fuse && nj >= 2 && jidx[1] == i + 1 && i + 2 < n_nodes) {
        ggml_tensor * t = cgraph->nodes[i + 2];
        if (t->op == GGML_OP_GLU && ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU && (t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            gj = t->src[0] == node ? 0 : t->src[0] == jobs[1].node ? 1 : -1;
            uj = t->src[1] == node ? 0 : t->src[1] == jobs[1].node ? 1 : -1;
            const int64_t F = inf0.h.rows;
            if (gj >= 0 && uj >= 0 && gj != uj && !jobs[1].deferred && jobs[1].inf.h.rows == F && !(inf0.h.flags & 2) && !(jobs[1].inf.h.flags & 2) &&
                    F % 512 == 0 && ytr_rows_f32(t, F, n_tok) && ytr_rows_f32(node, F, n_tok) && ytr_rows_f32(jobs[1].node, F, n_tok)) {
                glu = t; kind = 2; skip = 2;
            }
        }
    }
    const int64_t D = inf0.h.rows;
    if (!no_fuse && kind == 0 && (inf0.h.flags & 3) && D % 512 == 0 && D <= 8192 && ytr_rows_f32(node, D, n_tok)) {
        auto next_real = [&](int k) { while (k < n_nodes && ytr_noop(cgraph->nodes[k])) { ++k; } return k; };
        auto of_node = [&](const ggml_tensor * t) {
            return t == node || (t != nullptr && t->view_src == node && t->data == node->data && ggml_is_contiguous(t) && ytr_rows_f32(t, D, n_tok));
        };
        const int ka = next_real(i + 1);
        ggml_tensor * a = ka < n_nodes ? cgraph->nodes[ka] : nullptr;
        if (a && a->op == GGML_OP_ADD && (a->flags & GGML_TENSOR_FLAG_COMPUTE) && ytr_rows_f32(a, D, n_tok) && (of_node(a->src[0]) || of_node(a->src[1]))) {
            res = a->src[of_node(a->src[0]) ? 1 : 0];
            const int kr = next_real(ka + 1);
            ggml_tensor * r = kr < n_nodes ? cgraph->nodes[kr] : nullptr;
            const int km = r ? next_real(kr + 1) : n_nodes;
            ggml_tensor * m = km < n_nodes ? cgraph->nodes[km] : nullptr;
            if (ytr_rows_f32(res, D, n_tok) && r && r->op == GGML_OP_RMS_NORM && r->src[0] == a && (r->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    ytr_rows_f32(r, D, n_tok) && m && m->op == GGML_OP_MUL && (m->flags & GGML_TENSOR_FLAG_COMPUTE) && ytr_rows_f32(m, D, n_tok) &&
                    (m->src[0] == r || m->src[1] == r)) {
                const ggml_tensor * wt = m->src[0] == r ? m->src[1] : m->src[0];
                if (wt->op == GGML_OP_NONE && wt->type == GGML_TYPE_F32 && wt->ne[0] == D && ggml_nrows(wt) == 1 && ggml_is_contiguous(wt)) {
                    add = a; rms = r; mul = m; kind = 1; skip = km - i; ka_i = ka; kr_i = kr;
                }
            }
        }
    }

    uint64_t epoch;
    ytr_bufs * B = ytr_get_bufs(ctx.device, epoch);
    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    ytr_ensure_x8(B, x, node, inf0, epoch, stream);
    ytr_gemv(B, jobs, nj, n_tok, nsm, stream, x, sides, n_sides, sides_taken);

    // the consumer of the folded epilogue's int8 input
    ytr_info cinf;
    const ggml_tensor * cons = nullptr;
    if (kind == 1) {
        cons = ytr_next_product(cgraph, i + skip, mul, &cinf);
        if (cons && (cinf.h.cols != D)) {
            cons = nullptr;
        }
    } else if (kind == 2) {
        cons = ytr_next_product(cgraph, i + skip, glu, &cinf);
        if (cons && (cinf.h.cols != inf0.h.rows || (cinf.h.flags & 4))) {
            cons = nullptr;
        }
    }
    unsigned folded = 0;
    if (kind == 1) {
        folded = 1u;
    } else if (kind == 2) {
        folded = 3u;
    }
    {
        int last = i + skip;
        for (int s = 0; s < nj; ++s) {
            last = std::max(last, jidx[s]);
        }
        ytr_fin(jobs, nj, folded, n_tok, kind ? nullptr : ytr_next_product(cgraph, last, nullptr, nullptr), stream);
    }
    if (kind == 1) {
        const ytr_header & h = inf0.h;
        const uint8_t * base = (const uint8_t *) node->src[0]->data;
        ytr_e1_args E = {};
        E.acc = jobs[0].acc;
        E.sout = (h.flags & 1) ? base + h.sout_off : nullptr;
        E.rinv = inf0.rinv;
        // intermediate results only the folded nodes read are not written
        E.mm_out = ytr_read_only_by(cgraph, i, ka_i) ? nullptr : (float *) node->data;
        E.s_mm   = node->nb[1] / sizeof(float);
        E.res = (const float *) res->data;        E.s_res  = res->nb[1] / sizeof(float);
        E.add_out = (float *) add->data;          E.s_add  = add->nb[1] / sizeof(float);
        E.norm_out = ytr_read_only_by(cgraph, kr_i, i + skip) ? nullptr : (float *) rms->data;
        E.s_norm = rms->nb[1] / sizeof(float);
        E.w = (const float *) (mul->src[0] == rms ? mul->src[1] : mul->src[0])->data;
        E.mul_out = (float *) mul->data;          E.s_mul  = mul->nb[1] / sizeof(float);
        memcpy(&E.eps, rms->op_params, sizeof(float));
        E.D = (int) D;
        if (cons) {
            const uint8_t * cb = (const uint8_t *) cons->src[0]->data;
            E.sin = cb + cinf.h.sin_off;
            E.cmap = (cinf.h.flags & 4) ? (const int32_t *) (cb + cinf.h.cmap_off) : nullptr;
            E.x8 = B->x8;
            E.alpha = B->al;
        }
        ggml_cuda_kernel_launch(ytr_e1_kernel, ggml_cuda_kernel_launch_params(dim3(n_tok), dim3((unsigned) (D / 512) * 32), (size_t) (D + 32) * sizeof(float), stream), E);
        if (cons) {
            B->x_src = mul; B->x_key = cinf.in_key; B->x_epoch = epoch;
        }
    } else if (kind == 2) {
        const int64_t F = inf0.h.rows;
        const ytr_job & G = jobs[gj], & U = jobs[uj];
        const uint8_t * gb = (const uint8_t *) G.node->src[0]->data, * ub = (const uint8_t *) U.node->src[0]->data;
        ytr_e2_args E = {};
        E.acc_g = G.acc; E.acc_u = U.acc;
        E.sout_g = (G.inf.h.flags & 1) ? gb + G.inf.h.sout_off : nullptr;
        E.sout_u = (U.inf.h.flags & 1) ? ub + U.inf.h.sout_off : nullptr;
        // gate and up outputs only the SWIGLU reads are not written; neither is the SWIGLU output when the down product right after
        // it is its only reader (it takes the int8 input made here)
        E.g_out = ytr_read_only_by(cgraph, i + gj, i + 2) ? nullptr : (float *) G.node->data;
        E.u_out = ytr_read_only_by(cgraph, i + uj, i + 2) ? nullptr : (float *) U.node->data;
        E.s_g  = G.node->nb[1] / sizeof(float);
        E.s_u  = U.node->nb[1] / sizeof(float);
        int kd = i + 3;
        while (kd < cgraph->n_nodes && ytr_noop(cgraph->nodes[kd])) {
            ++kd;
        }
        const bool sw_x8_only = cons != nullptr && kd < cgraph->n_nodes && cgraph->nodes[kd] == cons && ytr_read_only_by(cgraph, i + 2, kd);
        E.sw_out = sw_x8_only ? nullptr : (float *) glu->data;
        E.s_sw = glu->nb[1] / sizeof(float);
        E.F = (int) F;
        if (cons) {
            E.sin = (const uint8_t *) cons->src[0]->data + cinf.h.sin_off;
            E.x8 = B->x8;
            E.alpha = B->al;
        }
        ggml_cuda_kernel_launch(ytr_e2_kernel, ggml_cuda_kernel_launch_params(dim3((unsigned) ((F / 512 + 3) / 4), n_tok), dim3(128), 0, stream), E);
        if (cons) {
            B->x_src = glu; B->x_key = cinf.in_key; B->x_epoch = epoch;
        }
    }

    // later products of the launch are done; node i too when nothing after it is skipped
    for (int s = 1; s < nj; ++s) {
        if (jobs[s].deferred) {
            std::lock_guard<std::mutex> lock(g_ytr_mutex);
            g_ytr_defer[jobs[s].node] = jobs[s];
            g_ytr_defer_n[jobs[s].node] = 1;
        } else if (jidx[s] > i + skip) {
            ctx.precomputed.insert(jobs[s].node);
        }
    }
    if (skip == 0) {
        std::lock_guard<std::mutex> lock(g_ytr_mutex);
        g_ytr_done.insert(node);
    }
    return skip;
}
