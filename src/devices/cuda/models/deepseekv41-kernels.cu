//
// DeepSeek-V4.1 专用 CUDA kernel。
//
// 数值语义与 src/devices/cpu/deepseekv41ops.cpp 中的 CPU 参考实现一致；
// 这里以正确性优先，面向 SM86 等无 FP8 tensor core 的设备，全部用 FP32 计算。
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

namespace {
    using fastllm::Data;
    using fastllm::DataType;
    using fastllm::DataDevice;

    // 在当前 CUDA 设备上准备输出张量（与 deepseekv4-kernels.cu 中的同名逻辑一致）
    bool V41PrepareOutput(Data &output, DataType dataType, const std::vector<int> &dims) {
        output.dataType = dataType;
        output.Resize(dims);
        output.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
        output.Allocate(false);
        return output.cudaData != nullptr;
    }

    // ---------------- 类型转换 ----------------

    template <typename T> __device__ __forceinline__ float V41Load(const T *p, uint64_t i);
    template <> __device__ __forceinline__ float V41Load<float>(const float *p, uint64_t i) { return p[i]; }
    template <> __device__ __forceinline__ float V41Load<__nv_bfloat16>(const __nv_bfloat16 *p, uint64_t i) { return __bfloat162float(p[i]); }
    template <> __device__ __forceinline__ float V41Load<half>(const half *p, uint64_t i) { return __half2float(p[i]); }

    template <typename T> __device__ __forceinline__ void V41Store(T *p, uint64_t i, float v);
    template <> __device__ __forceinline__ void V41Store<float>(float *p, uint64_t i, float v) { p[i] = v; }
    template <> __device__ __forceinline__ void V41Store<__nv_bfloat16>(__nv_bfloat16 *p, uint64_t i, float v) { p[i] = __float2bfloat16_rn(v); }
    template <> __device__ __forceinline__ void V41Store<half>(half *p, uint64_t i, float v) { p[i] = __float2half_rn(v); }

    __device__ __forceinline__ float V41Bf16Round(float v) {
        return __bfloat162float(__float2bfloat16_rn(v));
    }

    __device__ __forceinline__ float V41SigmoidDev(float x) {
        return 1.0f / (1.0f + __expf(-x));
    }

    // 2^ceil(log2(x))
    __device__ __forceinline__ float V41Pow2CeilDev(float x) {
        if (!(x > 0.0f)) {
            return 1.0f;
        }
        unsigned bits = __float_as_uint(x);
        int exponent = (int)((bits >> 23) & 0xFF) - 127 + ((bits & ((1u << 23) - 1)) != 0 ? 1 : 0);
        return ldexpf(1.0f, exponent);
    }

    __device__ __forceinline__ float V41Fp8RoundTripDev(float v) {
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(v);
        return (float)q;
    }

    __device__ __forceinline__ float V41Fp4RoundTripDev(float v) {
        const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float a = fabsf(v);
        if (a >= 6.0f) {
            return copysignf(6.0f, v);
        }
        int lower = 0;
#pragma unroll
        for (int i = 1; i < 8; i++) {
            if (grid[i] <= a) {
                lower = i;
            }
        }
        float lo = grid[lower];
        float hi = grid[lower + 1 < 8 ? lower + 1 : 7];
        float r;
        if (a == lo) {
            r = lo;
        } else {
            float mid = 0.5f * (lo + hi);
            if (a < mid) {
                r = lo;
            } else if (a > mid) {
                r = hi;
            } else {
                r = ((lower & 1) == 0) ? lo : hi;
            }
        }
        return copysignf(r, v);
    }

    __device__ __forceinline__ float V41QuantScaleDev(float amax, int quantMode) {
        if (quantMode == 1) {
            amax = fmaxf(amax, 1e-4f);
            return V41Pow2CeilDev(amax * (1.0f / 448.0f));
        } else if (quantMode == 2) {
            amax = fmaxf(amax, 6.0f * ldexpf(1.0f, -126));
            return V41Pow2CeilDev(amax * (1.0f / 6.0f));
        }
        amax = fmaxf(amax, 6.0f * ldexpf(1.0f, -9));
        float s = V41Fp8RoundTripDev(amax / 6.0f);
        return s > 0.0f ? s : ldexpf(1.0f, -9);
    }

    __device__ __forceinline__ float V41QuantValueDev(float v, float scale, int quantMode) {
        float qmax = quantMode == 1 ? 448.0f : 6.0f;
        float q = fminf(qmax, fmaxf(-qmax, v / scale));
        q = quantMode == 1 ? V41Fp8RoundTripDev(q) : V41Fp4RoundTripDev(q);
        return q * scale;
    }

    struct V41RopeTable {
        float invFreq[64];
        int pairs;
    };

    // 块内 warp 归约
    __device__ __forceinline__ float V41WarpSum(float v) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        return v;
    }

    __device__ __forceinline__ float V41WarpMax(float v) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        return v;
    }

    template <int THREADS>
    __device__ __forceinline__ float V41BlockSum(float v, float *shared) {
        v = V41WarpSum(v);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        __syncthreads();
        if (lane == 0) {
            shared[warp] = v;
        }
        __syncthreads();
        float total = 0.0f;
        for (int i = 0; i < THREADS / 32; i++) {
            total += shared[i];
        }
        return total;
    }

    template <int THREADS>
    __device__ __forceinline__ float V41BlockMax(float v, float *shared) {
        v = V41WarpMax(v);
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        __syncthreads();
        if (lane == 0) {
            shared[warp] = v;
        }
        __syncthreads();
        float total = -FLT_MAX;
        for (int i = 0; i < THREADS / 32; i++) {
            total = fmaxf(total, shared[i]);
        }
        return total;
    }

    // ---------------- HcMix ----------------

    constexpr int kHcThreads = 256;
    constexpr int kHcMaxMix = 32;   // (2 + hc) * hc, hc <= 4

    template <typename T>
    __global__ void V41HcMixKernel(const T *x, const float *fn, const float *scale, const float *base,
                                   int hcMult, int dim, int sinkhornIters, float eps, float normEps,
                                   float *pre, float *post, float *comb) {
        const int t = blockIdx.x;
        const int flatDim = hcMult * dim;
        const int mixHc = (2 + hcMult) * hcMult;
        __shared__ float sharedPartial[kHcMaxMix + 1][kHcThreads / 32];
        __shared__ float mixes[kHcMaxMix];
        __shared__ float combShared[16];
        float acc[kHcMaxMix + 1];
#pragma unroll
        for (int m = 0; m <= kHcMaxMix; m++) {
            acc[m] = 0.0f;
        }
        const T *xrow = x + (uint64_t)t * flatDim;
        for (int k = threadIdx.x; k < flatDim; k += kHcThreads) {
            float xv = V41Load<T>(xrow, k);
            acc[kHcMaxMix] += xv * xv;
            for (int m = 0; m < mixHc; m++) {
                acc[m] += xv * fn[(uint64_t)m * flatDim + k];
            }
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int m = 0; m <= mixHc; m++) {
            int idx = m == mixHc ? kHcMaxMix : m;
            float v = V41WarpSum(acc[idx]);
            if (lane == 0) {
                sharedPartial[idx][warp] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x <= mixHc) {
            int idx = threadIdx.x == mixHc ? kHcMaxMix : threadIdx.x;
            float total = 0.0f;
            for (int w = 0; w < kHcThreads / 32; w++) {
                total += sharedPartial[idx][w];
            }
            if (threadIdx.x == mixHc) {
                sharedPartial[kHcMaxMix][0] = total;   // sumsq
            } else {
                mixes[idx] = total;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float ss = sharedPartial[kHcMaxMix][0];
            float rsqrtv = rsqrtf(ss / flatDim + normEps);
            float *preOut = pre + (uint64_t)t * hcMult;
            float *postOut = post + (uint64_t)t * hcMult;
            float *combOut = comb + (uint64_t)t * hcMult * hcMult;
            for (int h = 0; h < hcMult; h++) {
                preOut[h] = V41SigmoidDev(mixes[h] * rsqrtv * scale[0] + base[h]) + eps;
                postOut[h] = 2.0f * V41SigmoidDev(mixes[h + hcMult] * rsqrtv * scale[1] + base[h + hcMult]);
            }
            for (int r = 0; r < hcMult; r++) {
                float rowMax = -FLT_MAX;
                for (int c = 0; c < hcMult; c++) {
                    int idx = r * hcMult + c + 2 * hcMult;
                    combShared[r * hcMult + c] = mixes[idx] * rsqrtv * scale[2] + base[idx];
                    rowMax = fmaxf(rowMax, combShared[r * hcMult + c]);
                }
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    float v = __expf(combShared[r * hcMult + c] - rowMax);
                    combShared[r * hcMult + c] = v;
                    rowSum += v;
                }
                for (int c = 0; c < hcMult; c++) {
                    combShared[r * hcMult + c] = combShared[r * hcMult + c] / rowSum + eps;
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combShared[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combShared[r * hcMult + c] /= (colSum + eps);
                }
            }
            for (int it = 1; it < sinkhornIters; it++) {
                for (int r = 0; r < hcMult; r++) {
                    float rowSum = 0.0f;
                    for (int c = 0; c < hcMult; c++) {
                        rowSum += combShared[r * hcMult + c];
                    }
                    for (int c = 0; c < hcMult; c++) {
                        combShared[r * hcMult + c] /= (rowSum + eps);
                    }
                }
                for (int c = 0; c < hcMult; c++) {
                    float colSum = 0.0f;
                    for (int r = 0; r < hcMult; r++) {
                        colSum += combShared[r * hcMult + c];
                    }
                    for (int r = 0; r < hcMult; r++) {
                        combShared[r * hcMult + c] /= (colSum + eps);
                    }
                }
            }
            for (int i = 0; i < hcMult * hcMult; i++) {
                combOut[i] = combShared[i];
            }
        }
    }

    // ---------------- HcApplyPre ----------------

    template <typename T>
    __global__ void V41HcApplyPreKernel(const T *x, const float *pre, T *y, int tokens, int hcMult, int dim) {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t total = (uint64_t)tokens * dim;
        if (idx >= total) {
            return;
        }
        int t = (int)(idx / dim), d = (int)(idx % dim);
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[(uint64_t)t * hcMult + h] * V41Load<T>(x, ((uint64_t)t * hcMult + h) * dim + d);
        }
        V41Store<T>(y, idx, v);
    }

    // ---------------- EngramApply ----------------

    constexpr int kEngramThreads = 256;

    template <typename HT, typename KT>
    __global__ void V41EngramApplyKernel(HT *hidden, const KT *kv, const float *qw, const float *kw,
                                         const float *mask, int hcMult, int dim, float eps, float clampValue) {
        const int t = blockIdx.x / hcMult;
        const int c = blockIdx.x % hcMult;
        __shared__ float shared[kEngramThreads / 32];
        HT *h = hidden + ((uint64_t)t * hcMult + c) * dim;
        const KT *key = kv + (uint64_t)t * (hcMult + 1) * dim + (uint64_t)c * dim;
        const KT *value = kv + (uint64_t)t * (hcMult + 1) * dim + (uint64_t)hcMult * dim;
        float hss = 0.0f, kss = 0.0f, dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += kEngramThreads) {
            float hv = V41Load<HT>(h, d);
            float kvv = V41Load<KT>(key, d);
            hss += hv * hv;
            kss += kvv * kvv;
            dot += hv * (qw[c * dim + d] * kw[c * dim + d]) * kvv;
        }
        hss = V41BlockSum<kEngramThreads>(hss, shared);
        kss = V41BlockSum<kEngramThreads>(kss, shared);
        dot = V41BlockSum<kEngramThreads>(dot, shared);
        float rstd = rsqrtf(hss / dim + eps) * rsqrtf(kss / dim + eps);
        float score = dot * rstd * rsqrtf((float)dim);
        float mag = sqrtf(fmaxf(fabsf(score), clampValue));
        float gate = V41SigmoidDev(copysignf(mag, score));
        if (mask != nullptr && mask[t] == 0.0f) {
            gate = 0.0f;
        }
        for (int d = threadIdx.x; d < dim; d += kEngramThreads) {
            float hv = V41Load<HT>(h, d);
            V41Store<HT>(h, d, hv + gate * V41Load<KT>(value, d));
        }
    }

    // ---------------- RotaryQuant ----------------

    // 一个 block 处理一行，blockDim = dim（<= 1024）
    template <typename T>
    __global__ void V41RotaryQuantKernel(T *x, int rowsPerToken, int dim, V41RopeTable rope,
                                         int ropeDim, int startPos, int posStep, int inverse,
                                         int quantMode, int quantDim, int quantBlock) {
        extern __shared__ float row[];
        const int r = blockIdx.x;
        const int token = r / rowsPerToken;
        T *base = x + (uint64_t)r * dim;
        const int d = threadIdx.x;
        row[d] = V41Load<T>(base, d);
        __syncthreads();
        const int off = dim - ropeDim;
        if (d < ropeDim / 2) {
            float pos = (float)(startPos + (long long)token * posStep);
            float ang = pos * rope.invFreq[d];
            float c = cosf(ang), s = sinf(ang);
            if (inverse) {
                s = -s;
            }
            float a = row[off + 2 * d], b = row[off + 2 * d + 1];
            row[off + 2 * d] = a * c - b * s;
            row[off + 2 * d + 1] = a * s + b * c;
        }
        __syncthreads();
        float v = row[d];
        if (quantMode > 0 && d < quantDim) {
            // 组内 amax：组是连续的 quantBlock（16 / 32）个元素，与 warp 对齐
            int group = d / quantBlock;
            float a = fabsf(v);
            unsigned mask = 0xffffffff;
            if (quantBlock == 16) {
                a = fmaxf(a, __shfl_xor_sync(mask, a, 8));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 4));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 2));
                a = fmaxf(a, __shfl_xor_sync(mask, a, 1));
            } else {
                a = V41WarpMax(a);
            }
            (void)group;
            float scale = V41QuantScaleDev(a, quantMode);
            v = V41QuantValueDev(v, scale, quantMode);
        }
        V41Store<T>(base, d, v);
    }

    // ---------------- Compress ----------------

    template <typename T>
    __global__ void V41CompressKernel(const T *kv, const T *score, const float *normWeight,
                                      int n, int dim, int ratio, float normEps, __nv_bfloat16 *out) {
        extern __shared__ float pooled[];
        __shared__ float shared[32];
        const int idx = blockIdx.x;       // b * blocks + j
        const int blocks = n / ratio;
        const int b = idx / blocks, j = idx % blocks;
        const int d = threadIdx.x;
        float value;
        if (ratio == 1) {
            value = V41Load<T>(kv, ((uint64_t)b * n + j) * dim + d);
        } else {
            float mx = -FLT_MAX;
            for (int r = 0; r < ratio; r++) {
                mx = fmaxf(mx, V41Load<T>(score, ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d));
            }
            float sum = 0.0f, acc = 0.0f;
            for (int r = 0; r < ratio; r++) {
                uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                float e = __expf(V41Load<T>(score, off) - mx);
                sum += e;
                acc += e * V41Load<T>(kv, off);
            }
            value = acc / sum;
        }
        value = V41Bf16Round(value);
        pooled[d] = value;
        float ss = V41BlockSum<512>(value * value, shared);
        float rsqrtv = rsqrtf(ss / dim + normEps);
        out[(uint64_t)idx * dim + d] = __float2bfloat16_rn(normWeight[d] * pooled[d] * rsqrtv);
    }

    // ---------------- IndexerScore ----------------

    constexpr int kIdxThreads = 128;

    template <typename QT, typename KT>
    __global__ void V41IndexerScoreKernel(const QT *q, const float *weights, const KT *k,
                                          int seqlen, int heads, int dim, int m, float *out) {
        extern __shared__ float qs[];   // heads * dim + heads
        const int t = blockIdx.y;       // b * seqlen + i
        const int b = t / seqlen;
        float *ws = qs + heads * dim;
        for (int i = threadIdx.x; i < heads * dim; i += kIdxThreads) {
            qs[i] = V41Load<QT>(q, (uint64_t)t * heads * dim + i);
        }
        for (int i = threadIdx.x; i < heads; i += kIdxThreads) {
            ws[i] = weights[(uint64_t)t * heads + i];
        }
        __syncthreads();
        const int j = blockIdx.x * kIdxThreads + threadIdx.x;
        if (j >= m) {
            return;
        }
        const KT *krow = k + ((uint64_t)b * m + j) * dim;
        float kvals[128];
        for (int d = 0; d < dim; d++) {
            kvals[d] = V41Load<KT>(krow, d);
        }
        float total = 0.0f;
        for (int h = 0; h < heads; h++) {
            const float *qh = qs + h * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += qh[d] * kvals[d];
            }
            total += fmaxf(dot, 0.0f) * ws[h];
        }
        out[(uint64_t)t * m + j] = total;
    }

    // ---------------- radix select（k-th largest） ----------------

    __device__ __forceinline__ unsigned V41FloatKey(float f) {
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }

    constexpr int kSelThreads = 256;

    // 返回第 k 大的 key（keys 由 getKey(i) 给出，i in [0, n)）。若 n <= k，返回 0。
    template <typename GetKey>
    __device__ unsigned V41RadixSelect(int n, int k, GetKey getKey, unsigned *hist /* shared, 16 */) {
        unsigned prefix = 0;
        int remaining = k;
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (threadIdx.x < 16) {
                hist[threadIdx.x] = 0;
            }
            __syncthreads();
            unsigned mask = shift == 28 ? 0u : (0xffffffffu << (shift + 4));
            for (int i = threadIdx.x; i < n; i += kSelThreads) {
                unsigned key = getKey(i);
                if ((key & mask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 15], 1u);
                }
            }
            __syncthreads();
            // 从高 digit 往低找
            if (threadIdx.x == 0) {
                int cnt = remaining;
                int digit = 15;
                for (; digit >= 0; digit--) {
                    int c = (int)hist[digit];
                    if (cnt <= c) {
                        break;
                    }
                    cnt -= c;
                }
                if (digit < 0) {
                    digit = 0;
                }
                hist[0] = (unsigned)digit;
                hist[1] = (unsigned)cnt;
            }
            __syncthreads();
            unsigned digit = hist[0];
            remaining = (int)hist[1];
            prefix |= digit << shift;
            __syncthreads();
        }
        return prefix;
    }

    // ---------------- CandidateBlocks ----------------

    __global__ void V41BlockScoreKernel(const float *score, int seqlen, int m, int blockSize, int numBlocks,
                                        int ratio, int startPos, float *blockScore) {
        const int t = blockIdx.x;
        const int i = t % seqlen;
        const int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        float *brow = blockScore + (uint64_t)t * numBlocks;
        for (int k = threadIdx.x; k < numBlocks; k += blockDim.x) {
            float mx = -INFINITY;
            int end = min(m, (k + 1) * blockSize);
            for (int j = k * blockSize; j < end; j++) {
                if (j < visible) {
                    mx = fmaxf(mx, row[j]);
                }
            }
            if (visible > 0 && k == (visible - 1) / blockSize) {
                mx = INFINITY;
            }
            brow[k] = mx;
        }
    }

    __global__ void V41CandidateSelectKernel(const float *blockScore, int numBlocks, int topkBlocks, uint8_t *mask) {
        const int t = blockIdx.x;
        const float *brow = blockScore + (uint64_t)t * numBlocks;
        uint8_t *mrow = mask + (uint64_t)t * numBlocks;
        __shared__ unsigned hist[16];
        int keep = min(topkBlocks, numBlocks);
        unsigned threshold = V41RadixSelect(numBlocks, keep,
            [&](int i) { return V41FloatKey(brow[i]); }, hist);
        const unsigned negInf = V41FloatKey(-INFINITY);
        // 严格大于的全部保留，等于阈值的按预算保留（顺序无关紧要，候选掩码只需数量约束）
        for (int i = threadIdx.x; i < numBlocks; i += kSelThreads) {
            unsigned key = V41FloatKey(brow[i]);
            uint8_t v = 0;
            if (key > threshold && key > negInf) {
                v = 1;
            }
            mrow[i] = v;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < numBlocks; i += kSelThreads) {
            unsigned key = V41FloatKey(brow[i]);
            if (key == threshold && key > negInf) {
                mrow[i] = 1;   // 阈值上的块全部保留（可能略多于 topkBlocks，只放宽候选范围）
            }
        }
    }

    // ---------------- IndexerTopK ----------------

    __global__ void V41TopKKernel(const float *score, const uint8_t *candidates, int seqlen, int m,
                                  int numBlocks, int blockSize, int topK, int width, int ratio, int startPos,
                                  int32_t *out) {
        typedef cub::BlockScan<int, kSelThreads> BlockScan;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ unsigned hist[16];
        const int t = blockIdx.x;
        const int i = t % seqlen;
        const int visible = min(m, (startPos + i + 1) / ratio);
        const float *row = score + (uint64_t)t * m;
        const uint8_t *cand = candidates == nullptr ? nullptr : candidates + (uint64_t)t * numBlocks;
        int32_t *orow = out + (uint64_t)t * width;

        auto eligible = [&](int j) -> bool {
            if (j >= visible) {
                return false;
            }
            if (cand != nullptr) {
                int blk = j / blockSize;
                if (blk >= numBlocks || cand[blk] == 0) {
                    return false;
                }
            }
            return true;
        };
        const unsigned negInf = V41FloatKey(-INFINITY);
        auto getKey = [&](int j) -> unsigned {
            return eligible(j) ? V41FloatKey(row[j]) : negInf;
        };

        // 统计可用数量
        int local = 0;
        for (int j = threadIdx.x; j < visible; j += kSelThreads) {
            local += eligible(j) ? 1 : 0;
        }
        int total = 0;
        BlockScan(scanStorage).ExclusiveSum(local, local, total);
        __syncthreads();
        int keep = min(width, total);
        unsigned threshold = 0;
        int tieBudget = 0;
        if (total > width) {
            threshold = V41RadixSelect(visible, width, getKey, hist);
            // 严格大于的数量
            int greater = 0;
            for (int j = threadIdx.x; j < visible; j += kSelThreads) {
                unsigned key = getKey(j);
                greater += (key > threshold && key > negInf) ? 1 : 0;
            }
            int greaterTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(greater, greater, greaterTotal);
            __syncthreads();
            tieBudget = width - greaterTotal;
        }
        // 按升序压缩写出
        int runningSel = 0, runningTie = 0;
        for (int st = 0; st < visible; st += kSelThreads) {
            int j = st + threadIdx.x;
            bool valid = j < visible;
            unsigned key = valid ? getKey(j) : negInf;
            int isTie = 0, isSel = 0;
            if (valid && key > negInf) {
                if (total <= width) {
                    isSel = 1;
                } else if (key > threshold) {
                    isSel = 1;
                } else if (key == threshold) {
                    isTie = 1;
                }
            }
            int tieRank = 0, tieTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(isTie, tieRank, tieTotal);
            __syncthreads();
            if (isTie && runningTie + tieRank < tieBudget) {
                isSel = 1;
            }
            int selRank = 0, selTotal = 0;
            BlockScan(scanStorage).ExclusiveSum(isSel, selRank, selTotal);
            __syncthreads();
            if (isSel) {
                int pos = runningSel + selRank;
                if (pos < width) {
                    orow[pos] = j;
                }
            }
            runningSel += selTotal;
            runningTie += tieTotal;
        }
        for (int p = keep + threadIdx.x; p < width; p += kSelThreads) {
            orow[p] = -1;
        }
    }

    // ---------------- SparseAttention ----------------

    constexpr int kAttnHeadsPerBlock = 32;
    constexpr int kAttnLanes = 8;
    constexpr int kAttnThreads = kAttnHeadsPerBlock * kAttnLanes;   // 256

    // FP8 缓存行：[dim 个 E4M3][dim / 32 个 UE8M0]
    __device__ __forceinline__ float V41LoadFp8Row(const uint8_t *row, int dim, int d) {
        __nv_fp8_e4m3 c;
        c.__x = row[d];
        return (float)c * ldexpf(1.0f, (int)row[dim + (d >> 5)] - 127);
    }

    template <typename QT>
    __global__ void __launch_bounds__(kAttnThreads)
    V41SparseAttentionKernel(const QT *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                             const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                             const int32_t *cmpIdx,
                             const float *sink, int seqlen, int heads, int dim,
                             int windowSize, int cap, int topWidth, int startPos,
                             float scale, __nv_bfloat16 *out) {
        __shared__ float kvRow[512];
        const int t = blockIdx.x;                  // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        constexpr int lanesPerHead = kAttnLanes;          // 8
        constexpr int segment = 512 / lanesPerHead;       // 64
        const int h = blockIdx.y * kAttnHeadsPerBlock + threadIdx.x / lanesPerHead;
        const int lane = threadIdx.x % lanesPerHead;
        const QT *qrow = q + ((uint64_t)t * heads + h) * dim + lane * segment;
        float qv[64];
        for (int d = 0; d < segment; d++) {
            qv[d] = V41Load<QT>(qrow, d);
        }
        float acc[64];
        for (int d = 0; d < segment; d++) {
            acc[d] = 0.0f;
        }
        float mx = sink[h];
        float l = 1.0f;

        const int winStart = max(0, pos - windowSize + 1);
        const int winCount = pos - winStart + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        const int rowBytes8 = dim + (dim >> 5);
        for (int c = 0; c < winCount + cmpCount; c++) {
            const __nv_bfloat16 *src = nullptr;
            const uint8_t *src8 = nullptr;
            if (c < winCount) {
                int p = winStart + c;
                if (p >= startPos) {
                    src = chunkKV + ((uint64_t)b * seqlen + (p - startPos)) * dim;
                } else if (ringKV8 != nullptr) {
                    src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * rowBytes8;
                } else {
                    src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * dim;
                }
            } else {
                int idx = idxRow[c - winCount];
                if (idx < 0 || idx >= cap) {
                    continue;
                }
                if (compressedKV8 != nullptr) {
                    src8 = compressedKV8 + ((uint64_t)b * cap + idx) * rowBytes8;
                } else {
                    src = compressedKV + ((uint64_t)b * cap + idx) * dim;
                }
            }
            __syncthreads();
            if (src8 != nullptr) {
                for (int d = threadIdx.x; d < dim; d += kAttnThreads) {
                    kvRow[d] = V41LoadFp8Row(src8, dim, d);
                }
            } else {
                for (int d = threadIdx.x; d < dim; d += kAttnThreads) {
                    kvRow[d] = __bfloat162float(src[d]);
                }
            }
            __syncthreads();
            float dot = 0.0f;
            const float *seg = kvRow + lane * segment;
            for (int d = 0; d < segment; d++) {
                dot += qv[d] * seg[d];
            }
            // 8 lane 归约
            dot += __shfl_xor_sync(0xffffffff, dot, 1);
            dot += __shfl_xor_sync(0xffffffff, dot, 2);
            dot += __shfl_xor_sync(0xffffffff, dot, 4);
            float s = dot * scale;
            float newMx = fmaxf(mx, s);
            float alpha = __expf(mx - newMx);
            float p = __expf(s - newMx);
            l = l * alpha + p;
            for (int d = 0; d < segment; d++) {
                acc[d] = acc[d] * alpha + p * seg[d];
            }
            mx = newMx;
        }
        float inv = 1.0f / l;
        __nv_bfloat16 *orow = out + ((uint64_t)t * heads + h) * dim + lane * segment;
        for (int d = 0; d < segment; d++) {
            orow[d] = __float2bfloat16_rn(acc[d] * inv);
        }
    }


    // ---------------- SparseAttention（BF16 mma 版本，SM80+） ----------------
    //
    // 数值语义与上面的 V41SparseAttentionKernel 一致（在线 softmax + attn_sink，
    // 候选顺序为「滑窗 -> 压缩 top-k」），区别只有：
    //   * QK^T 与 PV 用 mma.sync.m16n8k16（BF16 输入 / FP32 累加）代替 FP32 标量点积；
    //   * 候选按 kMmaNC 个一组分块，一组只要 4 次 __syncthreads（原来每个候选 1 次）；
    //   * 一个 block 内 32 个 head 共享同一份候选 KV，Q 也常驻共享内存；
    //   * 候选维可以再切成 gridDim.z 份（split-K），由 V41SparseMergeKernel 合并部分和，
    //     用于 decode 时提高并行度。
    //
    // 片段布局（PTX ISA 的 m16n8k16 定义）：
    //   A(16x16): lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1} [+8])
    //   B(16x8) : lane l 持有 (k = (l%4)*2 + {0,1} [+8], n = l/4)
    //   C(16x8) : lane l 持有 (row = l/4 [+8], col = (l%4)*2 + {0,1})

    constexpr int kMmaHeads = 32;                       // 每个 block 处理的 head 数
    constexpr int kMmaNC = 32;                          // 每轮处理的候选数
    constexpr int kMmaWarps = 8;
    constexpr int kMmaThreads = kMmaWarps * 32;         // 256
    constexpr int kMmaDim = 512;                        // V4.1 的 head_dim 固定为 512
    constexpr int kMmaKvStride = kMmaDim + 8;           // +8 个半字，消除 ldmatrix 的 bank 冲突
    constexpr int kMmaPStride = kMmaNC + 8;
    constexpr int kMmaDimSlice = kMmaDim / 4;           // PV 阶段每个 warp 负责 128 维
    constexpr int kMmaPvTiles = kMmaDimSlice / 8;       // 16 个 n-tile

    struct V41MmaShared {
        __nv_bfloat16 qs[kMmaHeads][kMmaKvStride];
        __nv_bfloat16 kvs[kMmaNC][kMmaKvStride];
        __nv_bfloat16 ps[kMmaHeads][kMmaPStride];
        float sc[kMmaHeads][kMmaNC];
        float mx[kMmaHeads];
        float lsum[kMmaHeads];
        float alpha[kMmaHeads];
        int valid[kMmaNC];
    };

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    __device__ __forceinline__ uint32_t V41SmemAddr(const void *p) {
        return static_cast<uint32_t>(__cvta_generic_to_shared(p));
    }

    __device__ __forceinline__ void V41LdmX4(uint32_t (&r)[4], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
    }

    __device__ __forceinline__ void V41LdmX2(uint32_t (&r)[2], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1]) : "r"(s));
    }

    __device__ __forceinline__ void V41LdmX2T(uint32_t (&r)[2], const void *addr) {
        uint32_t s = V41SmemAddr(addr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1]) : "r"(s));
    }

    __device__ __forceinline__ void V41MmaBf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#endif

    __global__ void __launch_bounds__(kMmaThreads)
    V41SparseAttentionMmaKernel(const __nv_bfloat16 *q, const __nv_bfloat16 *chunkKV, const __nv_bfloat16 *ringKV,
                                const __nv_bfloat16 *compressedKV, const uint8_t *ringKV8, const uint8_t *compressedKV8,
                                const int32_t *cmpIdx, const float *sink, int seqlen, int heads,
                                int windowSize, int cap, int topWidth, int startPos, float scale,
                                __nv_bfloat16 *out, float *partAcc, float *partMx, float *partL) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        extern __shared__ char v41MmaSharedRaw[];
        V41MmaShared &sh = *reinterpret_cast<V41MmaShared*>(v41MmaSharedRaw);

        const int t = blockIdx.x;                       // b * seqlen + i
        const int b = t / seqlen, i = t % seqlen;
        const int pos = startPos + i;
        const int h0 = blockIdx.y * kMmaHeads;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;

        for (int v = threadIdx.x; v < kMmaHeads * (kMmaDim / 8); v += kMmaThreads) {
            int hh = v / (kMmaDim / 8), dv = v % (kMmaDim / 8);
            *((float4*)&sh.qs[hh][dv * 8]) =
                *((const float4*)(q + ((uint64_t)t * heads + h0 + hh) * kMmaDim) + dv);
        }
        for (int r = threadIdx.x; r < kMmaHeads; r += kMmaThreads) {
            sh.mx[r] = blockIdx.z == 0 ? sink[h0 + r] : -FLT_MAX;
            sh.lsum[r] = blockIdx.z == 0 ? 1.0f : 0.0f;
        }

        const int winStart = max(0, pos - windowSize + 1);
        const int winCount = pos - winStart + 1;
        const int cmpCount = cmpIdx == nullptr ? 0 : topWidth;
        const int32_t *idxRow = cmpIdx == nullptr ? nullptr : cmpIdx + (uint64_t)t * topWidth;
        const int rowBytes8 = kMmaDim + (kMmaDim >> 5);
        const int totalCand = winCount + cmpCount;
        const int tilesTotal = (totalCand + kMmaNC - 1) / kMmaNC;
        const int tilesPerSplit = (tilesTotal + (int)gridDim.z - 1) / (int)gridDim.z;
        const int tileBegin = (int)blockIdx.z * tilesPerSplit;
        const int tileEnd = min(tilesTotal, tileBegin + tilesPerSplit);

        const int pvM = warp >> 2, pvSlice = warp & 3;   // PV: (mTile, 128 维切片)
        const int qkM = warp >> 2, qkN = warp & 3;       // QK: (mTile, 8 个候选)
        float acc[kMmaPvTiles][4];
#pragma unroll
        for (int n = 0; n < kMmaPvTiles; n++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[n][e] = 0.0f;
            }
        }
        const int qkARow = qkM * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int qkAColBlk = (lane >> 4) * 8;
        const int qkBRow = qkN * 8 + (lane & 7);
        const int qkBColBlk = ((lane >> 3) & 1) * 8;
        const int pvARow = pvM * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int pvAColBlk = (lane >> 4) * 8;
        const int pvBRow = lane & 15;

        for (int tile = tileBegin; tile < tileEnd; tile++) {
            const int tileStart = tile * kMmaNC;
            __syncthreads();
            for (int slot = warp; slot < kMmaNC; slot += kMmaWarps) {
                const int c = tileStart + slot;
                const __nv_bfloat16 *src = nullptr;
                const uint8_t *src8 = nullptr;
                bool ok = c < totalCand;
                if (ok) {
                    if (c < winCount) {
                        int p = winStart + c;
                        if (p >= startPos) {
                            src = chunkKV + ((uint64_t)b * seqlen + (p - startPos)) * kMmaDim;
                        } else if (ringKV8 != nullptr) {
                            src8 = ringKV8 + ((uint64_t)b * windowSize + (p % windowSize)) * rowBytes8;
                        } else {
                            src = ringKV + ((uint64_t)b * windowSize + (p % windowSize)) * kMmaDim;
                        }
                    } else {
                        int idx = idxRow[c - winCount];
                        if (idx < 0 || idx >= cap) {
                            ok = false;
                        } else if (compressedKV8 != nullptr) {
                            src8 = compressedKV8 + ((uint64_t)b * cap + idx) * rowBytes8;
                        } else {
                            src = compressedKV + ((uint64_t)b * cap + idx) * kMmaDim;
                        }
                    }
                }
                if (!ok) {
                    for (int v = lane; v < kMmaDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else if (src8 != nullptr) {
                    for (int d = lane; d < kMmaDim; d += 32) {
                        sh.kvs[slot][d] = __float2bfloat16_rn(V41LoadFp8Row(src8, kMmaDim, d));
                    }
                } else {
                    for (int v = lane; v < kMmaDim / 8; v += 32) {
                        *((float4*)&sh.kvs[slot][v * 8]) = *((const float4*)src + v);
                    }
                }
                if (lane == 0) {
                    sh.valid[slot] = ok ? 1 : 0;
                }
            }
            __syncthreads();

            {   // QK^T：C[16 head, 8 cand] = Q[16, 512] x KV[8, 512]^T
                float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint32_t a[4], bb[2];
#pragma unroll 2
                for (int k = 0; k < kMmaDim; k += 32) {
                    V41LdmX4(a, &sh.qs[qkARow][k + qkAColBlk]);
                    V41LdmX2(bb, &sh.kvs[qkBRow][k + qkBColBlk]);
                    V41MmaBf16(d0, a, bb);
                    V41LdmX4(a, &sh.qs[qkARow][k + 16 + qkAColBlk]);
                    V41LdmX2(bb, &sh.kvs[qkBRow][k + 16 + qkBColBlk]);
                    V41MmaBf16(d1, a, bb);
                }
                const int r = qkM * 16 + (lane >> 2);
                const int c = qkN * 8 + (lane & 3) * 2;
                sh.sc[r][c] = d0[0] + d1[0];
                sh.sc[r][c + 1] = d0[1] + d1[1];
                sh.sc[r + 8][c] = d0[2] + d1[2];
                sh.sc[r + 8][c + 1] = d0[3] + d1[3];
            }
            __syncthreads();

            {   // 在线 softmax：每个 head 一行，由 8 个线程负责
                constexpr int perThread = kMmaNC / 8;
                const int row = threadIdx.x >> 3;
                const int sub = threadIdx.x & 7;
                float vals[perThread];
                bool oks[perThread];
                float m = -FLT_MAX;
#pragma unroll
                for (int c = 0; c < perThread; c++) {
                    int col = sub * perThread + c;
                    bool ok = sh.valid[col] != 0;
                    float v = ok ? sh.sc[row][col] * scale : -FLT_MAX;
                    oks[c] = ok;
                    vals[c] = v;
                    m = fmaxf(m, v);
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
                const float oldMx = sh.mx[row];
                const float newMx = fmaxf(oldMx, m);
                float s = 0.0f;
#pragma unroll
                for (int c = 0; c < perThread; c++) {
                    float p = oks[c] ? __expf(vals[c] - newMx) : 0.0f;
                    s += p;
                    sh.ps[row][sub * perThread + c] = __float2bfloat16_rn(p);
                }
                s += __shfl_xor_sync(0xffffffff, s, 1);
                s += __shfl_xor_sync(0xffffffff, s, 2);
                s += __shfl_xor_sync(0xffffffff, s, 4);
                if (sub == 0) {
                    float a = __expf(oldMx - newMx);
                    sh.alpha[row] = a;
                    sh.mx[row] = newMx;
                    sh.lsum[row] = sh.lsum[row] * a + s;
                }
            }
            __syncthreads();

            {   // PV：O[16 head, 128 dim] = alpha * O + P[16, 32] x V[32, 128]
                const float a0 = sh.alpha[pvM * 16 + (lane >> 2)];
                const float a1 = sh.alpha[pvM * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int n = 0; n < kMmaPvTiles; n++) {
                    acc[n][0] *= a0;
                    acc[n][1] *= a0;
                    acc[n][2] *= a1;
                    acc[n][3] *= a1;
                }
#pragma unroll
                for (int k = 0; k < kMmaNC; k += 16) {
                    uint32_t a[4];
                    V41LdmX4(a, &sh.ps[pvARow][k + pvAColBlk]);
#pragma unroll
                    for (int n = 0; n < kMmaPvTiles; n++) {
                        uint32_t bb[2];
                        V41LdmX2T(bb, &sh.kvs[k + pvBRow][pvSlice * kMmaDimSlice + n * 8]);
                        V41MmaBf16(acc[n], a, bb);
                    }
                }
            }
        }
        __syncthreads();

        const int outRow0 = pvM * 16 + (lane >> 2);
        const int outRow1 = outRow0 + 8;
        const int outCol = pvSlice * kMmaDimSlice + (lane & 3) * 2;
        if (partAcc == nullptr) {
            const float inv0 = 1.0f / sh.lsum[outRow0];
            const float inv1 = 1.0f / sh.lsum[outRow1];
            __nv_bfloat16 *o0 = out + ((uint64_t)t * heads + h0 + outRow0) * kMmaDim;
            __nv_bfloat16 *o1 = out + ((uint64_t)t * heads + h0 + outRow1) * kMmaDim;
#pragma unroll
            for (int n = 0; n < kMmaPvTiles; n++) {
                int d = outCol + n * 8;
                o0[d] = __float2bfloat16_rn(acc[n][0] * inv0);
                o0[d + 1] = __float2bfloat16_rn(acc[n][1] * inv0);
                o1[d] = __float2bfloat16_rn(acc[n][2] * inv1);
                o1[d + 1] = __float2bfloat16_rn(acc[n][3] * inv1);
            }
        } else {
            const uint64_t base = ((uint64_t)blockIdx.z * gridDim.x + t) * heads;
            float *a0 = partAcc + (base + h0 + outRow0) * kMmaDim;
            float *a1 = partAcc + (base + h0 + outRow1) * kMmaDim;
#pragma unroll
            for (int n = 0; n < kMmaPvTiles; n++) {
                int d = outCol + n * 8;
                a0[d] = acc[n][0];
                a0[d + 1] = acc[n][1];
                a1[d] = acc[n][2];
                a1[d + 1] = acc[n][3];
            }
            if (warp == 0) {
                for (int r = lane; r < kMmaHeads; r += 32) {
                    partMx[base + h0 + r] = sh.mx[r];
                    partL[base + h0 + r] = sh.lsum[r];
                }
            }
        }
#endif
    }

    // split-K 的部分和合并（在线 softmax 的标准合并公式）
    __global__ void V41SparseMergeKernel(const float *partAcc, const float *partMx, const float *partL,
                                         int tokens, int heads, int splits, __nv_bfloat16 *out) {
        const int idx = blockIdx.x;                     // t * heads + h
        const uint64_t stride = (uint64_t)tokens * heads;
        float mx = -FLT_MAX;
        for (int s = 0; s < splits; s++) {
            mx = fmaxf(mx, partMx[(uint64_t)s * stride + idx]);
        }
        float denom = 0.0f;
        for (int s = 0; s < splits; s++) {
            denom += __expf(partMx[(uint64_t)s * stride + idx] - mx) * partL[(uint64_t)s * stride + idx];
        }
        const float inv = 1.0f / denom;
        for (int d = threadIdx.x; d < kMmaDim; d += blockDim.x) {
            float v = 0.0f;
            for (int s = 0; s < splits; s++) {
                v += __expf(partMx[(uint64_t)s * stride + idx] - mx) *
                     partAcc[((uint64_t)s * stride + idx) * kMmaDim + d];
            }
            out[(uint64_t)idx * kMmaDim + d] = __float2bfloat16_rn(v * inv);
        }
    }

    // 设备是否支持 BF16 mma（SM80+），按设备号缓存
    bool V41MmaSupported() {
        static int cache[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        int dev = FastllmCudaGetDevice();
        if (dev < 0 || dev >= 16) {
            return false;
        }
        if (cache[dev] < 0) {
            int major = 0;
            cache[dev] = (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          major >= 8) ? 1 : 0;
        }
        return cache[dev] == 1;
    }

    int V41SmCount() {
        static int cache[16] = {0};
        int dev = FastllmCudaGetDevice();
        if (dev < 0 || dev >= 16) {
            return 1;
        }
        if (cache[dev] == 0) {
            int n = 0;
            cache[dev] = (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0)
                         ? n : 1;
        }
        return cache[dev];
    }

    bool V41EnvOn(const char *name) {
        const char *v = getenv(name);
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }

    // ---------------- QuantizeKV ----------------

    constexpr int kKvQuantThreads = 128;

    // 每个 block 处理一行；每个 warp 处理 32 个一组的块，lane 对应块内元素
    template <typename T>
    __global__ void V41QuantizeKVKernel(const T *input, uint8_t *output, int rows, int dim) {
        const int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int nblocks = dim >> 5;
        const T *src = input + (uint64_t)row * dim;
        uint8_t *dst = output + (uint64_t)row * (dim + nblocks);
        uint8_t *scales = dst + dim;
        for (int blk = warp; blk < nblocks; blk += kKvQuantThreads / 32) {
            float v = V41Load<T>(src, blk * 32 + lane);
            float amax = V41WarpMax(fabsf(v));
            float scale = V41Pow2CeilDev(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
            float q = fminf(448.0f, fmaxf(-448.0f, v / scale));
            __nv_fp8_e4m3 code(q);
            dst[blk * 32 + lane] = code.__x;
            if (lane == 0) {
                int e = (int)((__float_as_uint(scale) >> 23) & 0xFF);   // scale 为正的 2 的幂
                scales[blk] = (uint8_t)e;
            }
        }
    }

    // ---------------- WindowStore ----------------

    __global__ void V41WindowStoreKernel(const uint8_t *chunk, uint8_t *ring, int seqlen, int rowBytes,
                                         int startPos, int windowSize, int firstRow) {
        const int b = blockIdx.y;
        const int i = firstRow + blockIdx.x;
        if (i >= seqlen) {
            return;
        }
        const int slot = (startPos + i) % windowSize;
        const uint8_t *src = chunk + ((uint64_t)b * seqlen + i) * rowBytes;
        uint8_t *dst = ring + ((uint64_t)b * windowSize + slot) * rowBytes;
        for (int k = threadIdx.x; k < rowBytes; k += blockDim.x) {
            dst[k] = src[k];
        }
    }

    // ---------------- host 辅助 ----------------

    bool V41OnCuda(const Data &d) {
        return d.dataDevice == DataDevice::CUDA && d.cudaData != nullptr;
    }

    bool V41IsFloatType(DataType t) {
        return t == DataType::FLOAT32 || t == DataType::FLOAT16 || t == DataType::BFLOAT16;
    }

    V41RopeTable V41BuildRope(int ropeDim, float base, int originalSeqLen, float factor, int betaFast, int betaSlow) {
        V41RopeTable table;
        table.pairs = ropeDim / 2;
        std::vector<float> invFreq;
        for (int i = 0; i < ropeDim; i += 2) {
            invFreq.push_back(1.0f / std::pow(base, (float)i / ropeDim));
        }
        if (originalSeqLen > 0) {
            auto correctedDim = [&](float rotations) {
                return ropeDim * std::log((float)originalSeqLen / (rotations * 2.0f * (float)M_PI)) /
                       (2.0f * std::log(base));
            };
            int low = std::max((int)std::floor(correctedDim((float)betaFast)), 0);
            int high = std::min((int)std::ceil(correctedDim((float)betaSlow)), ropeDim - 1);
            float denom = std::max((float)(high - low), 1e-3f);
            for (int i = 0; i < (int)invFreq.size(); i++) {
                float ramp = std::max(0.0f, std::min(1.0f, ((float)i - low) / denom));
                float smooth = 1.0f - ramp;
                invFreq[i] = invFreq[i] / factor * (1.0f - smooth) + invFreq[i] * smooth;
            }
        }
        for (int i = 0; i < 64; i++) {
            table.invFreq[i] = i < (int)invFreq.size() ? invFreq[i] : 0.0f;
        }
        return table;
    }

    bool V41CheckLaunch(const char *name) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[Fastllm] DeepSeekV41 CUDA kernel %s failed: %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
}

// ==================== 导出接口 ====================

extern "C" bool FastllmCudaDeepSeekV41HcMix(const fastllm::Data &x, fastllm::Data &hcFn, fastllm::Data &hcScale,
                                            fastllm::Data &hcBase, int hcMult, int sinkhornIters, float eps,
                                            float normEps, fastllm::Data &pre, fastllm::Data &post,
                                            fastllm::Data &comb) {
    if (!V41OnCuda(x) || x.dims.size() != 4 || hcMult <= 0 || hcMult > 4 || !V41IsFloatType(x.dataType) ||
        hcFn.dataType != DataType::FLOAT32 || hcScale.dataType != DataType::FLOAT32 ||
        hcBase.dataType != DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(pre, DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !V41PrepareOutput(post, DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !V41PrepareOutput(comb, DataType::FLOAT32, {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }
    hcFn.ToDevice(DataDevice::CUDA);
    hcScale.ToDevice(DataDevice::CUDA);
    hcBase.ToDevice(DataDevice::CUDA);
    if (x.dataType == DataType::BFLOAT16) {
        V41HcMixKernel<__nv_bfloat16><<<tokens, kHcThreads>>>(
            (const __nv_bfloat16*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    } else if (x.dataType == DataType::FLOAT16) {
        V41HcMixKernel<half><<<tokens, kHcThreads>>>(
            (const half*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    } else {
        V41HcMixKernel<float><<<tokens, kHcThreads>>>(
            (const float*)x.cudaData, (const float*)hcFn.cudaData, (const float*)hcScale.cudaData,
            (const float*)hcBase.cudaData, hcMult, dim, sinkhornIters, eps, normEps,
            (float*)pre.cudaData, (float*)post.cudaData, (float*)comb.cudaData);
    }
    return V41CheckLaunch("HcMix");
}

extern "C" bool FastllmCudaDeepSeekV41HcApplyPre(const fastllm::Data &x, const fastllm::Data &pre, fastllm::Data &y) {
    if (!V41OnCuda(x) || !V41OnCuda(pre) || x.dims.size() != 4 || pre.dataType != DataType::FLOAT32 ||
        !V41IsFloatType(x.dataType)) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], hcMult = x.dims[2], dim = x.dims[3];
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(y, x.dataType, {bsz, seqlen, dim})) {
        return false;
    }
    uint64_t total = (uint64_t)tokens * dim;
    int threads = 256;
    unsigned blocks = (unsigned)((total + threads - 1) / threads);
    if (x.dataType == DataType::BFLOAT16) {
        V41HcApplyPreKernel<__nv_bfloat16><<<blocks, threads>>>((const __nv_bfloat16*)x.cudaData, (const float*)pre.cudaData,
                                                                (__nv_bfloat16*)y.cudaData, tokens, hcMult, dim);
    } else if (x.dataType == DataType::FLOAT16) {
        V41HcApplyPreKernel<half><<<blocks, threads>>>((const half*)x.cudaData, (const float*)pre.cudaData,
                                                       (half*)y.cudaData, tokens, hcMult, dim);
    } else {
        V41HcApplyPreKernel<float><<<blocks, threads>>>((const float*)x.cudaData, (const float*)pre.cudaData,
                                                        (float*)y.cudaData, tokens, hcMult, dim);
    }
    return V41CheckLaunch("HcApplyPre");
}

extern "C" bool FastllmCudaDeepSeekV41EngramApply(fastllm::Data &hidden, const fastllm::Data &kv,
                                                  fastllm::Data &qWeight, fastllm::Data &kWeight,
                                                  const fastllm::Data *mask, float eps, float clampValue) {
    if (!V41OnCuda(hidden) || !V41OnCuda(kv) || hidden.dims.size() != 4 ||
        hidden.dataType != DataType::BFLOAT16 || !V41IsFloatType(kv.dataType) ||
        qWeight.dataType != DataType::FLOAT32 || kWeight.dataType != DataType::FLOAT32) {
        return false;
    }
    int bsz = hidden.dims[0], seqlen = hidden.dims[1], hcMult = hidden.dims[2], dim = hidden.dims[3];
    int tokens = bsz * seqlen;
    qWeight.ToDevice(DataDevice::CUDA);
    kWeight.ToDevice(DataDevice::CUDA);
    const float *maskPtr = nullptr;
    if (mask != nullptr && mask->Count(0) > 0) {
        if (!V41OnCuda(*mask) || mask->dataType != DataType::FLOAT32) {
            return false;
        }
        maskPtr = (const float*)mask->cudaData;
    }
    dim3 grid(tokens * hcMult);
    if (kv.dataType == DataType::BFLOAT16) {
        V41EngramApplyKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const __nv_bfloat16*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    } else if (kv.dataType == DataType::FLOAT16) {
        V41EngramApplyKernel<__nv_bfloat16, half><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const half*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    } else {
        V41EngramApplyKernel<__nv_bfloat16, float><<<grid, kEngramThreads>>>(
            (__nv_bfloat16*)hidden.cudaData, (const float*)kv.cudaData, (const float*)qWeight.cudaData,
            (const float*)kWeight.cudaData, maskPtr, hcMult, dim, eps, clampValue);
    }
    return V41CheckLaunch("EngramApply");
}

extern "C" bool FastllmCudaDeepSeekV41RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                                  int posStep, bool inverse, int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, int quantMode, int quantDim,
                                                  int quantBlock) {
    if (!V41OnCuda(x) || (x.dims.size() != 3 && x.dims.size() != 4) || !V41IsFloatType(x.dataType)) {
        return false;
    }
    int dim = x.dims.back();
    if (quantDim <= 0) {
        quantDim = dim;
    }
    if (dim > 1024 || dim % 32 != 0 || ropeDim <= 0 || ropeDim > 128 || ropeDim > dim ||
        (quantMode > 0 && ((quantBlock != 16 && quantBlock != 32) || quantDim % 32 != 0))) {
        return false;
    }
    int rowsPerToken = x.dims.size() == 4 ? x.dims[2] : 1;
    int rows = (int)(x.Count(0) / dim);
    V41RopeTable rope = V41BuildRope(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    size_t shared = (size_t)dim * sizeof(float);
    if (x.dataType == DataType::BFLOAT16) {
        V41RotaryQuantKernel<__nv_bfloat16><<<rows, dim, shared>>>((__nv_bfloat16*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
    } else if (x.dataType == DataType::FLOAT16) {
        V41RotaryQuantKernel<half><<<rows, dim, shared>>>((half*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
    } else {
        V41RotaryQuantKernel<float><<<rows, dim, shared>>>((float*)x.cudaData, rowsPerToken, dim, rope,
            ropeDim, startPos, posStep, inverse ? 1 : 0, quantMode, quantDim, quantBlock);
    }
    return V41CheckLaunch("RotaryQuant");
}

extern "C" bool FastllmCudaDeepSeekV41Compress(const fastllm::Data &kv, const fastllm::Data *score,
                                               fastllm::Data &normWeight, int ratio, float normEps,
                                               fastllm::Data &output) {
    if (!V41OnCuda(kv) || kv.dims.size() != 3 || kv.dims[2] != 512 || ratio <= 0 || kv.dims[1] % ratio != 0 ||
        !V41IsFloatType(kv.dataType) || (ratio > 1 && (score == nullptr || !V41OnCuda(*score) ||
                                                        score->dataType != kv.dataType))) {
        return false;
    }
    int bsz = kv.dims[0], n = kv.dims[1], dim = kv.dims[2];
    int blocks = n / ratio;
    normWeight.ToDevice(DataDevice::CUDA);
    if (normWeight.dataType != DataType::FLOAT32) {
        return false;
    }
    if (!V41PrepareOutput(output, DataType::BFLOAT16, {bsz, blocks, dim})) {
        return false;
    }
    size_t shared = (size_t)dim * sizeof(float);
    const void *scorePtr = ratio > 1 ? score->cudaData : nullptr;
    if (kv.dataType == DataType::FLOAT32) {
        V41CompressKernel<float><<<bsz * blocks, dim, shared>>>((const float*)kv.cudaData, (const float*)scorePtr,
            (const float*)normWeight.cudaData, n, dim, ratio, normEps, (__nv_bfloat16*)output.cudaData);
    } else if (kv.dataType == DataType::BFLOAT16) {
        V41CompressKernel<__nv_bfloat16><<<bsz * blocks, dim, shared>>>((const __nv_bfloat16*)kv.cudaData,
            (const __nv_bfloat16*)scorePtr, (const float*)normWeight.cudaData, n, dim, ratio, normEps,
            (__nv_bfloat16*)output.cudaData);
    } else {
        V41CompressKernel<half><<<bsz * blocks, dim, shared>>>((const half*)kv.cudaData, (const half*)scorePtr,
            (const float*)normWeight.cudaData, n, dim, ratio, normEps, (__nv_bfloat16*)output.cudaData);
    }
    return V41CheckLaunch("Compress");
}

extern "C" bool FastllmCudaDeepSeekV41IndexerScore(const fastllm::Data &q, const fastllm::Data &weights,
                                                   const fastllm::Data &k, fastllm::Data &output) {
    if (!V41OnCuda(q) || !V41OnCuda(weights) || !V41OnCuda(k) || q.dims.size() != 4 || k.dims.size() != 3 ||
        q.dims[3] != 128 || k.dims[2] != 128 || weights.dataType != DataType::FLOAT32 ||
        (q.dataType != DataType::BFLOAT16 && q.dataType != DataType::FLOAT32) ||
        (k.dataType != DataType::BFLOAT16 && k.dataType != DataType::FLOAT32)) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int m = k.dims[1];
    if (!V41PrepareOutput(output, DataType::FLOAT32, {bsz, seqlen, m})) {
        return false;
    }
    if (m == 0) {
        return true;
    }
    dim3 grid((m + kIdxThreads - 1) / kIdxThreads, bsz * seqlen);
    size_t shared = (size_t)(heads * dim + heads) * sizeof(float);
    if (q.dataType == DataType::BFLOAT16 && k.dataType == DataType::BFLOAT16) {
        V41IndexerScoreKernel<__nv_bfloat16, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
            (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData, (const __nv_bfloat16*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else if (q.dataType == DataType::FLOAT32 && k.dataType == DataType::BFLOAT16) {
        V41IndexerScoreKernel<float, __nv_bfloat16><<<grid, kIdxThreads, shared>>>(
            (const float*)q.cudaData, (const float*)weights.cudaData, (const __nv_bfloat16*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else if (q.dataType == DataType::BFLOAT16 && k.dataType == DataType::FLOAT32) {
        V41IndexerScoreKernel<__nv_bfloat16, float><<<grid, kIdxThreads, shared>>>(
            (const __nv_bfloat16*)q.cudaData, (const float*)weights.cudaData, (const float*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    } else {
        V41IndexerScoreKernel<float, float><<<grid, kIdxThreads, shared>>>(
            (const float*)q.cudaData, (const float*)weights.cudaData, (const float*)k.cudaData,
            seqlen, heads, dim, m, (float*)output.cudaData);
    }
    return V41CheckLaunch("IndexerScore");
}

extern "C" bool FastllmCudaDeepSeekV41CandidateBlocks(const fastllm::Data &score, int blockSize, int topkBlocks,
                                                      int ratio, int startPos, fastllm::Data &output) {
    if (!V41OnCuda(score) || score.dims.size() != 3 || score.dataType != DataType::FLOAT32 || blockSize <= 0) {
        return false;
    }
    int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
    int numBlocks = (m + blockSize - 1) / blockSize;
    int tokens = bsz * seqlen;
    if (!V41PrepareOutput(output, DataType::INT8, {bsz, seqlen, numBlocks})) {
        return false;
    }
    float *blockScore = (float*)FastllmCudaMalloc((size_t)tokens * numBlocks * sizeof(float));
    if (blockScore == nullptr) {
        return false;
    }
    V41BlockScoreKernel<<<tokens, 256>>>((const float*)score.cudaData, seqlen, m, blockSize, numBlocks,
                                         ratio, startPos, blockScore);
    V41CandidateSelectKernel<<<tokens, kSelThreads>>>(blockScore, numBlocks, topkBlocks, (uint8_t*)output.cudaData);
    bool ok = V41CheckLaunch("CandidateBlocks");
    cudaDeviceSynchronize();
    FastllmCudaFree(blockScore);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV41IndexerTopK(const fastllm::Data &score, const fastllm::Data *candidates,
                                                  int topK, int ratio, int startPos, int blockSize,
                                                  fastllm::Data &output) {
    if (!V41OnCuda(score) || score.dims.size() != 3 || score.dataType != DataType::FLOAT32 || topK <= 0) {
        return false;
    }
    int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
    int width = std::min(topK, m);
    const uint8_t *cand = nullptr;
    int numBlocks = 0;
    if (candidates != nullptr && candidates->Count(0) > 0) {
        if (!V41OnCuda(*candidates) || candidates->dataType != DataType::INT8 || candidates->dims.size() != 3) {
            return false;
        }
        cand = (const uint8_t*)candidates->cudaData;
        numBlocks = candidates->dims[2];
    }
    if (!V41PrepareOutput(output, DataType::INT32, {bsz, seqlen, width})) {
        return false;
    }
    if (width == 0) {
        return true;
    }
    V41TopKKernel<<<bsz * seqlen, kSelThreads>>>((const float*)score.cudaData, cand, seqlen, m, numBlocks,
                                                 std::max(1, blockSize), topK, width, ratio, startPos,
                                                 (int32_t*)output.cudaData);
    return V41CheckLaunch("IndexerTopK");
}

extern "C" bool FastllmCudaDeepSeekV41SparseAttention(const fastllm::Data &q, const fastllm::Data &chunkKV,
                                                      const fastllm::Data *ringKV, const fastllm::Data *compressedKV,
                                                      const fastllm::Data *cmpIdx, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, float softmaxScale,
                                                      fastllm::Data &output) {
    if (!V41OnCuda(q) || !V41OnCuda(chunkKV) || q.dims.size() != 4 || chunkKV.dims.size() != 3 ||
        q.dims[3] != 512 || q.dims[2] % kAttnHeadsPerBlock != 0 || chunkKV.dataType != DataType::BFLOAT16 ||
        (q.dataType != DataType::BFLOAT16 && q.dataType != DataType::FLOAT32)) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    bool hasRing = ringKV != nullptr && ringKV->dims.size() == 3 && ringKV->Count(0) > 0;
    bool hasCmp = compressedKV != nullptr && cmpIdx != nullptr && compressedKV->dims.size() == 3 &&
                  compressedKV->Count(0) > 0 && cmpIdx->dims.size() == 3;
    const int rowBytes8 = dim + dim / 32;
    bool ringFp8 = hasRing && ringKV->dataType == DataType::INT8;
    bool cmpFp8 = hasCmp && compressedKV->dataType == DataType::INT8;
    if (hasRing && (!V41OnCuda(*ringKV) || ringKV->dims[1] != windowSize ||
                    !(ringFp8 ? ringKV->dims[2] == rowBytes8
                              : (ringKV->dataType == DataType::BFLOAT16 && ringKV->dims[2] == dim)))) {
        return false;
    }
    if (hasCmp && (!V41OnCuda(*compressedKV) || !V41OnCuda(*cmpIdx) || cmpIdx->dataType != DataType::INT32 ||
                   !(cmpFp8 ? compressedKV->dims[2] == rowBytes8
                            : (compressedKV->dataType == DataType::BFLOAT16 && compressedKV->dims[2] == dim)))) {
        return false;
    }
    if (!hasRing && startPos > 0) {
        return false;
    }
    attnSink.ToDevice(DataDevice::CUDA);
    if (attnSink.dataType != DataType::FLOAT32) {
        return false;
    }
    if (!V41PrepareOutput(output, DataType::BFLOAT16, q.dims)) {
        return false;
    }
    int cap = hasCmp ? compressedKV->dims[1] : 0;
    int topWidth = hasCmp ? cmpIdx->dims[2] : 0;

    // BF16 mma 快速路径（SM80+，head_dim 512，q 为 BF16）。
    // FASTLLM_DSV41_LEGACY_ATTN=1 可退回下面的 FP32 标量 kernel 做对比 / 排查。
    if (q.dataType == DataType::BFLOAT16 && dim == kMmaDim && heads % kMmaHeads == 0 &&
        V41MmaSupported() && !V41EnvOn("FASTLLM_DSV41_LEGACY_ATTN")) {
        static bool sharedReady = false;
        if (!sharedReady) {
            cudaFuncSetAttribute(V41SparseAttentionMmaKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)sizeof(V41MmaShared));
            sharedReady = true;
        }
        const int tokens = bsz * seqlen;
        const int headBlocks = heads / kMmaHeads;
        const int maxCand = std::min(windowSize, startPos + seqlen) + topWidth;
        const int maxTiles = std::max(1, (maxCand + kMmaNC - 1) / kMmaNC);
        int splits = 1;
        const char *splitEnv = getenv("FASTLLM_DSV41_ATTN_SPLITS");
        if (splitEnv != nullptr && splitEnv[0] != '\0') {
            splits = std::max(1, std::min(atoi(splitEnv), maxTiles));
        } else {
            // decode（token 数少）时候选维 split-K，把 block 数补到约 4 倍 SM 数
            const int target = 4 * V41SmCount();
            const long long base = (long long)tokens * headBlocks;
            if (base > 0 && base < target && maxTiles > 1) {
                splits = (int)std::min((long long)maxTiles, (target + base - 1) / base);
                splits = std::max(1, std::min(splits, 32));
            }
        }
        float *partAcc = nullptr, *partMx = nullptr, *partL = nullptr;
        size_t partCount = 0;
        if (splits > 1) {
            partCount = (size_t)splits * tokens * heads;
            partAcc = (float*)FastllmCudaMalloc(partCount * (kMmaDim + 2) * sizeof(float));
            if (partAcc == nullptr) {
                splits = 1;
            } else {
                partMx = partAcc + partCount * kMmaDim;
                partL = partMx + partCount;
            }
        }
        dim3 grid(tokens, headBlocks, splits);
        V41SparseAttentionMmaKernel<<<grid, kMmaThreads, sizeof(V41MmaShared)>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData, partAcc, partMx, partL);
        if (splits > 1) {
            V41SparseMergeKernel<<<tokens * heads, 128>>>(partAcc, partMx, partL, tokens, heads, splits,
                                                          (__nv_bfloat16*)output.cudaData);
            FastllmCudaFree(partAcc);
        }
        return V41CheckLaunch("SparseAttentionMma");
    }

    if (q.dataType == DataType::BFLOAT16) {
        dim3 grid(bsz * seqlen, heads / kAttnHeadsPerBlock);
        V41SparseAttentionKernel<__nv_bfloat16><<<grid, kAttnThreads>>>(
            (const __nv_bfloat16*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, dim, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData);
    } else {
        dim3 grid(bsz * seqlen, heads / kAttnHeadsPerBlock);
        V41SparseAttentionKernel<float><<<grid, kAttnThreads>>>(
            (const float*)q.cudaData, (const __nv_bfloat16*)chunkKV.cudaData,
            hasRing && !ringFp8 ? (const __nv_bfloat16*)ringKV->cudaData : nullptr,
            hasCmp && !cmpFp8 ? (const __nv_bfloat16*)compressedKV->cudaData : nullptr,
            ringFp8 ? (const uint8_t*)ringKV->cudaData : nullptr,
            cmpFp8 ? (const uint8_t*)compressedKV->cudaData : nullptr,
            hasCmp ? (const int32_t*)cmpIdx->cudaData : nullptr,
            (const float*)attnSink.cudaData, seqlen, heads, dim, windowSize, cap, topWidth, startPos,
            softmaxScale, (__nv_bfloat16*)output.cudaData);
    }
    return V41CheckLaunch("SparseAttention");
}

extern "C" bool FastllmCudaDeepSeekV41WindowStore(const fastllm::Data &chunk, fastllm::Data &ring, int startPos,
                                                  int windowSize) {
    if (!V41OnCuda(chunk) || chunk.dims.size() != 3) {
        return false;
    }
    int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
    if (ring.dims.size() != 3 || ring.dims[0] != bsz || ring.dims[1] != windowSize || ring.dims[2] != dim ||
        ring.dataType != chunk.dataType || !V41OnCuda(ring)) {
        if (!V41PrepareOutput(ring, chunk.dataType, {bsz, windowSize, dim})) {
            return false;
        }
        cudaMemset(ring.cudaData, 0, (size_t)ring.Count(0) * ring.unitSize / ring.unitSizeDiv);
    }
    int rowBytes = dim * chunk.unitSize;
    int firstRow = std::max(0, seqlen - windowSize);
    dim3 grid(seqlen - firstRow, bsz);
    V41WindowStoreKernel<<<grid, 256>>>((const uint8_t*)chunk.cudaData, (uint8_t*)ring.cudaData, seqlen, rowBytes,
                                        startPos, windowSize, firstRow);
    return V41CheckLaunch("WindowStore");
}

extern "C" bool FastllmCudaDeepSeekV41QuantizeKV(const fastllm::Data &input, fastllm::Data &output) {
    if (!V41OnCuda(input) || input.dims.size() != 3 || input.dims[2] % 32 != 0 || !V41IsFloatType(input.dataType)) {
        return false;
    }
    int rows = input.dims[0] * input.dims[1], dim = input.dims[2];
    if (!V41PrepareOutput(output, DataType::INT8, {input.dims[0], input.dims[1], dim + dim / 32})) {
        return false;
    }
    if (rows == 0) {
        return true;
    }
    if (input.dataType == DataType::BFLOAT16) {
        V41QuantizeKVKernel<__nv_bfloat16><<<rows, kKvQuantThreads>>>(
            (const __nv_bfloat16*)input.cudaData, (uint8_t*)output.cudaData, rows, dim);
    } else if (input.dataType == DataType::FLOAT16) {
        V41QuantizeKVKernel<half><<<rows, kKvQuantThreads>>>(
            (const half*)input.cudaData, (uint8_t*)output.cudaData, rows, dim);
    } else {
        V41QuantizeKVKernel<float><<<rows, kKvQuantThreads>>>(
            (const float*)input.cudaData, (uint8_t*)output.cudaData, rows, dim);
    }
    return V41CheckLaunch("QuantizeKV");
}
