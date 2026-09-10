//
// DeepSeek-V4.1 专用算子的 CPU 参考实现。
//
// 这些算子同时作为 CUDA 实现的数值参考。所有算子均以 FP32 计算，
// 输入/输出支持 FLOAT32 / FLOAT16 / BFLOAT16。
//
//   DeepSeekV41HcMix           : Hyper-Connections 系数（pre / post / comb），不做混合
//   DeepSeekV41HcApplyPre      : 用给定 pre 系数把 hc 份隐藏状态折叠成一份
//   DeepSeekV41EngramApply     : Engram 门控写回 residual stream（原地）
//   DeepSeekV41RotaryQuant     : RoPE（可逆向）+ 可选 FP8 / FP4 伪量化（原地）
//   DeepSeekV41Compress        : compressor 池化（ratio 1 / 2）+ RMSNorm，输出 pre-RoPE latent
//   DeepSeekV41IndexerScore    : indexer 打分 score[t, j] = sum_h relu(q_h . k_j) * w_h
//   DeepSeekV41CandidateBlocks : 两级 top-k 的第一级（按块选候选）
//   DeepSeekV41IndexerTopK     : 因果 / 候选掩码后的 top-k（输出升序，-1 表示无效）
//   DeepSeekV41SparseAttention : 滑窗 + 压缩 top-k 的稀疏注意力（MQA，含 attention sink）
//   DeepSeekV41WindowStore     : 把当前 chunk 的 KV 写入环形滑窗缓存
//

#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"
#include "utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace fastllm {
    namespace {
        int V41Int(const IntDict &params, const char *name, int fallback) {
            auto it = params.find(name);
            return it == params.end() ? fallback : it->second;
        }

        float V41Float(const FloatDict &params, const char *name, float fallback) {
            auto it = params.find(name);
            return it == params.end() ? fallback : it->second;
        }

        Data *V41Optional(const DataDict &datas, const char *name) {
            auto it = datas.find(name);
            if (it == datas.end() || it->second == nullptr) {
                return nullptr;
            }
            return it->second;
        }

        inline float V41ToFloat(const uint8_t *base, DataType type, uint64_t index) {
            if (type == DataType::FLOAT32) {
                return ((const float*)base)[index];
            } else if (type == DataType::BFLOAT16) {
                return BFloat16BitsToFloat32(((const uint16_t*)base)[index]);
            } else if (type == DataType::FLOAT16) {
                return half_to_float(((const uint16_t*)base)[index]);
            }
            return 0.0f;
        }

        inline void V41FromFloat(uint8_t *base, DataType type, uint64_t index, float value) {
            if (type == DataType::FLOAT32) {
                ((float*)base)[index] = value;
            } else if (type == DataType::BFLOAT16) {
                ((uint16_t*)base)[index] = Float32ToBFloat16RNEBits(value);
            } else if (type == DataType::FLOAT16) {
                ((uint16_t*)base)[index] = float_to_half(value);
            }
        }

        bool V41IsFloatType(DataType type) {
            return type == DataType::FLOAT32 || type == DataType::BFLOAT16 || type == DataType::FLOAT16;
        }

        void V41ReadFloat(const Data &data, std::vector<float> &out) {
            uint64_t n = data.Count(0);
            out.resize(n);
            if (data.dataType == DataType::FLOAT32) {
                memcpy(out.data(), data.cpuData, n * sizeof(float));
                return;
            }
            for (uint64_t i = 0; i < n; i++) {
                out[i] = V41ToFloat(data.cpuData, data.dataType, i);
            }
        }

        void V41WriteFloat(const std::vector<float> &values, Data &data) {
            uint64_t n = data.Count(0);
            if (data.dataType == DataType::FLOAT32) {
                memcpy(data.cpuData, values.data(), n * sizeof(float));
                return;
            }
            for (uint64_t i = 0; i < n; i++) {
                V41FromFloat(data.cpuData, data.dataType, i, values[i]);
            }
        }

        // 简单的多线程并行 for：把 [0, count) 切成若干段
        struct V41RangeTask : MultiThreadBaseOp {
            std::function<void(int, int)> fn;
            int st, end;
            V41RangeTask(std::function<void(int, int)> fn, int st, int end) : fn(std::move(fn)), st(st), end(end) {}
            void Run() override { fn(st, end); }
        };

        void V41ParallelFor(int count, const std::function<void(int, int)> &fn, int minPerThread = 1) {
            if (count <= 0) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            int firstThread = pool->curActivateThreadInterval.first;
            int available = std::max(1, pool->curActivateThreadInterval.second - firstThread);
            int threads = std::min(available, std::max(1, count / std::max(1, minPerThread)));
            if (threads <= 1) {
                fn(0, count);
                return;
            }
            std::vector<V41RangeTask*> tasks;
            int per = (count + threads - 1) / threads;
            for (int i = 0; i < threads; i++) {
                int st = i * per, end = std::min(count, st + per);
                if (st >= end) {
                    break;
                }
                tasks.push_back(new V41RangeTask(fn, st, end));
            }
            for (int i = 0; i < (int)tasks.size(); i++) {
                pool->PushOp(firstThread + i, tasks[i]);
            }
            for (int i = 0; i < (int)tasks.size(); i++) {
                pool->Wait(firstThread + i);
                delete tasks[i];
            }
        }

        inline float V41Sigmoid(float x) {
            if (x >= 0.0f) {
                return 1.0f / (1.0f + std::exp(-x));
            }
            float z = std::exp(x);
            return z / (1.0f + z);
        }

        // 与 model.py::precompute_freqs_cis 一致的 YaRN 频率
        std::vector<float> V41InvFreq(int ropeDim, float base, int originalSeqLen,
                                      float factor, int betaFast, int betaSlow) {
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
            return invFreq;
        }
    }

    // ---------------- FP8 / FP4 伪量化 ----------------

    static float V41Pow2Ceil(float x) {
        // fast_round_scale: 2^ceil(log2(x))
        if (!(x > 0.0f)) {
            return 1.0f;
        }
        uint32_t bits;
        memcpy(&bits, &x, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xFF) - 127 + ((bits & ((1u << 23) - 1)) != 0 ? 1 : 0);
        return std::ldexp(1.0f, exponent);
    }

    static float V41FP8RoundTrip(float v) {
        static const FP8E4M3ToFP32Manager fp8;
        return fp8.dict[fp8.quantization(v)];
    }

    static float V41FP4RoundTrip(float v) {
        // E2M1 (无 inf / nan)：0, 0.5, 1, 1.5, 2, 3, 4, 6，RNE
        static const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float a = std::fabs(v);
        if (a >= 6.0f) {
            return std::copysign(6.0f, v);
        }
        int lower = 0;
        while (lower + 1 < 8 && grid[lower + 1] <= a) {
            lower++;
        }
        float lo = grid[lower], hi = grid[std::min(lower + 1, 7)];
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
                r = ((lower & 1) == 0) ? lo : hi; // 平局取偶数编码
            }
        }
        return std::copysign(r, v);
    }

    // quantMode: 1 = FP8 E4M3 + UE8M0 scale（act_quant, 每 blockSize 一组）
    //            2 = FP4 E2M1 + UE8M0 scale（fp4_act_quant scale_dtype=e8m0）
    //            3 = FP4 E2M1 + E4M3 scale（fp4_act_quant scale_dtype=e4m3, 压缩 KV）
    void DeepSeekV41FakeQuantRow(float *row, int len, int quantMode, int blockSize) {
        if (quantMode <= 0) {
            return;
        }
        for (int start = 0; start < len; start += blockSize) {
            int end = std::min(start + blockSize, len);
            float amax = 0.0f;
            for (int i = start; i < end; i++) {
                amax = std::max(amax, std::fabs(row[i]));
            }
            float scale;
            float qmax;
            if (quantMode == 1) {
                qmax = 448.0f;
                amax = std::max(amax, 1e-4f);
                scale = V41Pow2Ceil(amax * (1.0f / qmax));
            } else if (quantMode == 2) {
                qmax = 6.0f;
                amax = std::max(amax, 6.0f * std::ldexp(1.0f, -126));
                scale = V41Pow2Ceil(amax * (1.0f / qmax));
            } else {
                qmax = 6.0f;
                amax = std::max(amax, 6.0f * std::ldexp(1.0f, -9));
                scale = V41FP8RoundTrip(amax / qmax);
                if (!(scale > 0.0f)) {
                    scale = std::ldexp(1.0f, -9);
                }
            }
            for (int i = start; i < end; i++) {
                float q = std::max(-qmax, std::min(qmax, row[i] / scale));
                if (quantMode == 1) {
                    q = V41FP8RoundTrip(q);
                } else {
                    q = V41FP4RoundTrip(q);
                }
                row[i] = q * scale;
            }
        }
    }

    // ---------------- HcMix ----------------

    void CpuDeepSeekV41HcMixOp::Reshape(const std::string &opType, const DataDict &datas,
                                        const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        int hcMult = V41Int(intParams, "hcMult", 1);
        AssertInFastLLM(input.dims.size() == 4 && input.dims[2] == hcMult,
                        "DeepSeekV41HcMix error: input should be [b, s, hc, d].\n");
        int bsz = input.dims[0], seqlen = input.dims[1];
        pre.dataType = DataType::FLOAT32;
        pre.Resize({bsz, seqlen, hcMult});
        post.dataType = DataType::FLOAT32;
        post.Resize({bsz, seqlen, hcMult});
        comb.dataType = DataType::FLOAT32;
        comb.Resize({bsz, seqlen, hcMult, hcMult});
    }

    // 单 token 的 mixes -> pre / post / comb（与 kernel.py::hc_split_sinkhorn 一致）
    void DeepSeekV41SplitSinkhorn(const float *mixes, const float *scale, const float *base,
                                  int hcMult, int sinkhornIters, float eps,
                                  float *pre, float *post, float *comb) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = V41Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[h] = 2.0f * V41Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }
        for (int r = 0; r < hcMult; r++) {
            float rowMax = -FLT_MAX;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c + 2 * hcMult;
                comb[r * hcMult + c] = mixes[idx] * scale[2] + base[idx];
                rowMax = std::max(rowMax, comb[r * hcMult + c]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                float v = std::exp(comb[r * hcMult + c] - rowMax);
                comb[r * hcMult + c] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                comb[r * hcMult + c] = comb[r * hcMult + c] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += comb[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                comb[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += comb[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    comb[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += comb[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    comb[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
    }

    void CpuDeepSeekV41HcMixOp::Run(const std::string &opType, const DataDict &datas,
                                    const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &hcFn = *(datas.find("hcFn")->second);
        Data &hcScale = *(datas.find("hcScale")->second);
        Data &hcBase = *(datas.find("hcBase")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        int hcMult = V41Int(intParams, "hcMult", 1);
        int sinkhornIters = V41Int(intParams, "sinkhornIters", 20);
        float eps = V41Float(floatParams, "eps", 1e-6f);
        float normEps = V41Float(floatParams, "normEps", 1e-6f);

        int bsz = input.dims[0], seqlen = input.dims[1], dim = input.dims[3];
        int tokens = bsz * seqlen;
        int flatDim = hcMult * dim;
        int mixHc = (2 + hcMult) * hcMult;
        AssertInFastLLM(V41IsFloatType(input.dataType) && V41IsFloatType(hcFn.dataType) &&
                        hcScale.dataType == DataType::FLOAT32 && hcBase.dataType == DataType::FLOAT32 &&
                        hcFn.Count(0) == (uint64_t)mixHc * flatDim && hcScale.Count(0) >= 3 &&
                        hcBase.Count(0) >= (uint64_t)mixHc,
                        "DeepSeekV41HcMix error: invalid inputs.\n");
        std::vector<float> fnStorage;
        const float *fn;
        if (hcFn.dataType == DataType::FLOAT32) {
            fn = (const float*)hcFn.cpuData;
        } else {
            V41ReadFloat(hcFn, fnStorage);
            fn = fnStorage.data();
        }
        const float *scale = (const float*)hcScale.cpuData;
        const float *base = (const float*)hcBase.cpuData;
        pre.Allocate();
        post.Allocate();
        comb.Allocate();
        float *preData = (float*)pre.cpuData;
        float *postData = (float*)post.cpuData;
        float *combData = (float*)comb.cpuData;

        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> x(flatDim), mixes(mixHc);
            for (int t = st; t < end; t++) {
                for (int k = 0; k < flatDim; k++) {
                    x[k] = V41ToFloat(input.cpuData, input.dataType, (uint64_t)t * flatDim + k);
                }
                double ss = 0.0;
                for (int k = 0; k < flatDim; k++) {
                    ss += (double)x[k] * x[k];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / flatDim) + normEps);
                for (int m = 0; m < mixHc; m++) {
                    const float *w = fn + (uint64_t)m * flatDim;
                    double v = 0.0;
                    for (int k = 0; k < flatDim; k++) {
                        v += (double)x[k] * w[k];
                    }
                    mixes[m] = (float)v * rsqrt;
                }
                DeepSeekV41SplitSinkhorn(mixes.data(), scale, base, hcMult, sinkhornIters, eps,
                                         preData + (uint64_t)t * hcMult,
                                         postData + (uint64_t)t * hcMult,
                                         combData + (uint64_t)t * hcMult * hcMult);
            }
        }, 4);
    }

    // ---------------- HcApplyPre ----------------

    void CpuDeepSeekV41HcApplyPreOp::Reshape(const std::string &opType, const DataDict &datas,
                                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(input.dims.size() == 4, "DeepSeekV41HcApplyPre error: input should be [b, s, hc, d].\n");
        output.dataType = input.dataType;
        output.Resize({input.dims[0], input.dims[1], input.dims[3]});
    }

    void CpuDeepSeekV41HcApplyPreOp::Run(const std::string &opType, const DataDict &datas,
                                         const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &output = *(datas.find("output")->second);
        int bsz = input.dims[0], seqlen = input.dims[1], hcMult = input.dims[2], dim = input.dims[3];
        int tokens = bsz * seqlen;
        AssertInFastLLM(pre.dataType == DataType::FLOAT32 && pre.Count(0) == (uint64_t)tokens * hcMult,
                        "DeepSeekV41HcApplyPre error: invalid pre.\n");
        output.Allocate();
        const float *preData = (const float*)pre.cpuData;
        V41ParallelFor(tokens, [&](int st, int end) {
            for (int t = st; t < end; t++) {
                for (int d = 0; d < dim; d++) {
                    float v = 0.0f;
                    for (int h = 0; h < hcMult; h++) {
                        v += preData[(uint64_t)t * hcMult + h] *
                             V41ToFloat(input.cpuData, input.dataType, ((uint64_t)t * hcMult + h) * dim + d);
                    }
                    V41FromFloat(output.cpuData, output.dataType, (uint64_t)t * dim + d, v);
                }
            }
        }, 8);
    }

    // ---------------- EngramApply ----------------

    void CpuDeepSeekV41EngramApplyOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &hidden = *(datas.find("hidden")->second);
        Data &kv = *(datas.find("kv")->second);
        Data &qWeight = *(datas.find("qWeight")->second);
        Data &kWeight = *(datas.find("kWeight")->second);
        Data *mask = V41Optional(datas, "mask");
        float eps = V41Float(floatParams, "eps", 1e-20f);
        float clampValue = V41Float(floatParams, "clampValue", 1e-6f);

        AssertInFastLLM(hidden.dims.size() == 4, "DeepSeekV41EngramApply error: hidden should be [b, s, hc, d].\n");
        int bsz = hidden.dims[0], seqlen = hidden.dims[1], hcMult = hidden.dims[2], dim = hidden.dims[3];
        int tokens = bsz * seqlen;
        AssertInFastLLM(kv.Count(0) == (uint64_t)tokens * (hcMult + 1) * dim &&
                        qWeight.Count(0) == (uint64_t)hcMult * dim && kWeight.Count(0) == (uint64_t)hcMult * dim &&
                        qWeight.dataType == DataType::FLOAT32 && kWeight.dataType == DataType::FLOAT32,
                        "DeepSeekV41EngramApply error: shape mismatch.\n");
        const float *qw = (const float*)qWeight.cpuData;
        const float *kw = (const float*)kWeight.cpuData;
        std::vector<float> maskValues;
        if (mask != nullptr) {
            V41ReadFloat(*mask, maskValues);
        }
        uint64_t kvStride = (uint64_t)(hcMult + 1) * dim;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> h(dim), key(dim), value(dim);
            for (int t = st; t < end; t++) {
                for (int d = 0; d < dim; d++) {
                    value[d] = V41ToFloat(kv.cpuData, kv.dataType, (uint64_t)t * kvStride + (uint64_t)hcMult * dim + d);
                }
                for (int c = 0; c < hcMult; c++) {
                    uint64_t hOff = ((uint64_t)t * hcMult + c) * dim;
                    double hss = 0.0, kss = 0.0, dot = 0.0;
                    for (int d = 0; d < dim; d++) {
                        h[d] = V41ToFloat(hidden.cpuData, hidden.dataType, hOff + d);
                        key[d] = V41ToFloat(kv.cpuData, kv.dataType, (uint64_t)t * kvStride + (uint64_t)c * dim + d);
                        hss += (double)h[d] * h[d];
                        kss += (double)key[d] * key[d];
                        dot += (double)h[d] * (qw[c * dim + d] * kw[c * dim + d]) * key[d];
                    }
                    float rstd = (1.0f / std::sqrt((float)(hss / dim) + eps)) *
                                 (1.0f / std::sqrt((float)(kss / dim) + eps));
                    float score = (float)dot * rstd * (1.0f / std::sqrt((float)dim));
                    float mag = std::sqrt(std::max(std::fabs(score), clampValue));
                    float gate = V41Sigmoid(std::copysign(mag, score));
                    if (mask != nullptr && maskValues[t] == 0.0f) {
                        gate = 0.0f;
                    }
                    for (int d = 0; d < dim; d++) {
                        V41FromFloat(hidden.cpuData, hidden.dataType, hOff + d, h[d] + gate * value[d]);
                    }
                }
            }
        }, 4);
    }

    // ---------------- RotaryQuant ----------------

    void DeepSeekV41RotaryQuantRows(float *values, int rows, int rowsPerToken, int dim,
                                    int ropeDim, float ropeBase, int startPos, int posStep, bool inverse,
                                    int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                    int quantMode, int quantDim, int quantBlock) {
        auto invFreq = V41InvFreq(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        int pairs = ropeDim / 2;
        int off = dim - ropeDim;
        V41ParallelFor(rows, [&](int st, int end) {
            for (int r = st; r < end; r++) {
                int token = r / rowsPerToken;
                float pos = (float)(startPos + (int64_t)token * posStep);
                float *row = values + (uint64_t)r * dim + off;
                for (int p = 0; p < pairs; p++) {
                    float ang = pos * invFreq[p];
                    float c = std::cos(ang), s = std::sin(ang);
                    if (inverse) {
                        s = -s;
                    }
                    float a = row[2 * p], b = row[2 * p + 1];
                    row[2 * p] = a * c - b * s;
                    row[2 * p + 1] = a * s + b * c;
                }
                if (quantMode > 0) {
                    DeepSeekV41FakeQuantRow(values + (uint64_t)r * dim, quantDim, quantMode, quantBlock);
                }
            }
        }, 16);
    }

    void CpuDeepSeekV41RotaryQuantOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int ropeDim = V41Int(intParams, "ropeDim", 64);
        int startPos = V41Int(intParams, "startPos", 0);
        int posStep = V41Int(intParams, "posStep", 1);
        bool inverse = V41Int(intParams, "inverse", 0) != 0;
        int originalSeqLen = V41Int(intParams, "originalSeqLen", 0);
        int betaFast = V41Int(intParams, "betaFast", 32);
        int betaSlow = V41Int(intParams, "betaSlow", 1);
        int quantMode = V41Int(intParams, "quantMode", 0);
        int quantBlock = V41Int(intParams, "quantBlock", 32);
        float ropeBase = V41Float(floatParams, "ropeBase", 10000.0f);
        float ropeFactor = V41Float(floatParams, "ropeFactor", 1.0f);

        AssertInFastLLM(input.dims.size() == 3 || input.dims.size() == 4,
                        "DeepSeekV41RotaryQuant error: input should be [b, s, d] or [b, s, h, d].\n");
        int dim = input.dims.back();
        int quantDim = V41Int(intParams, "quantDim", dim);
        int rowsPerToken = input.dims.size() == 4 ? input.dims[2] : 1;
        int rows = (int)(input.Count(0) / dim);
        AssertInFastLLM(ropeDim > 0 && ropeDim <= dim && ropeDim % 2 == 0 && quantDim <= dim,
                        "DeepSeekV41RotaryQuant error: invalid params.\n");
        std::vector<float> values;
        V41ReadFloat(input, values);
        DeepSeekV41RotaryQuantRows(values.data(), rows, rowsPerToken, dim, ropeDim, ropeBase, startPos, posStep,
                                   inverse, originalSeqLen, ropeFactor, betaFast, betaSlow,
                                   quantMode, quantDim, quantBlock);
        V41WriteFloat(values, input);
    }

    // ---------------- Compress ----------------

    void CpuDeepSeekV41CompressOp::Reshape(const std::string &opType, const DataDict &datas,
                                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &kv = *(datas.find("kv")->second);
        Data &output = *(datas.find("output")->second);
        int ratio = V41Int(intParams, "compressRatio", 1);
        AssertInFastLLM(kv.dims.size() == 3 && ratio > 0 && kv.dims[1] % ratio == 0,
                        "DeepSeekV41Compress error: kv should be [b, n, d] with n % ratio == 0.\n");
        output.dataType = DataType::BFLOAT16;
        output.Resize({kv.dims[0], kv.dims[1] / ratio, kv.dims[2]});
    }

    void CpuDeepSeekV41CompressOp::Run(const std::string &opType, const DataDict &datas,
                                       const FloatDict &floatParams, const IntDict &intParams) {
        Data &kv = *(datas.find("kv")->second);
        Data *score = V41Optional(datas, "score");
        Data &normWeight = *(datas.find("normWeight")->second);
        Data &output = *(datas.find("output")->second);
        int ratio = V41Int(intParams, "compressRatio", 1);
        float normEps = V41Float(floatParams, "normEps", 1e-20f);
        int bsz = kv.dims[0], n = kv.dims[1], dim = kv.dims[2];
        int blocks = n / ratio;
        AssertInFastLLM(ratio == 1 || (score != nullptr && score->dims == kv.dims),
                        "DeepSeekV41Compress error: score is required when ratio > 1.\n");
        std::vector<float> normValues;
        V41ReadFloat(normWeight, normValues);
        AssertInFastLLM((int)normValues.size() >= dim, "DeepSeekV41Compress error: norm weight mismatch.\n");
        output.Allocate();
        int total = bsz * blocks;
        V41ParallelFor(total, [&](int st, int end) {
            std::vector<float> pooled(dim);
            for (int idx = st; idx < end; idx++) {
                int b = idx / blocks, j = idx % blocks;
                for (int d = 0; d < dim; d++) {
                    if (ratio == 1) {
                        pooled[d] = V41ToFloat(kv.cpuData, kv.dataType, ((uint64_t)b * n + j) * dim + d);
                    } else {
                        float mx = -FLT_MAX;
                        for (int r = 0; r < ratio; r++) {
                            uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                            mx = std::max(mx, V41ToFloat(score->cpuData, score->dataType, off));
                        }
                        double sum = 0.0, value = 0.0;
                        for (int r = 0; r < ratio; r++) {
                            uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                            double e = std::exp((double)V41ToFloat(score->cpuData, score->dataType, off) - mx);
                            sum += e;
                            value += e * V41ToFloat(kv.cpuData, kv.dataType, off);
                        }
                        pooled[d] = (float)(value / sum);
                    }
                    // kv.to(dtype): 归一化前先转 BF16
                    pooled[d] = BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(pooled[d]));
                }
                double ss = 0.0;
                for (int d = 0; d < dim; d++) {
                    ss += (double)pooled[d] * pooled[d];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / dim) + normEps);
                for (int d = 0; d < dim; d++) {
                    ((uint16_t*)output.cpuData)[(uint64_t)idx * dim + d] =
                        Float32ToBFloat16RNEBits(normValues[d] * pooled[d] * rsqrt);
                }
            }
        }, 4);
    }

    // ---------------- IndexerScore ----------------

    void CpuDeepSeekV41IndexerScoreOp::Reshape(const std::string &opType, const DataDict &datas,
                                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(q.dims.size() == 4 && k.dims.size() == 3 && q.dims[3] == k.dims[2] && q.dims[0] == k.dims[0],
                        "DeepSeekV41IndexerScore error: q should be [b, s, h, d], k should be [b, m, d].\n");
        output.dataType = DataType::FLOAT32;
        output.Resize({q.dims[0], q.dims[1], k.dims[1]});
    }

    void CpuDeepSeekV41IndexerScoreOp::Run(const std::string &opType, const DataDict &datas,
                                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &weights = *(datas.find("weights")->second);
        Data &k = *(datas.find("k")->second);
        Data &output = *(datas.find("output")->second);
        int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
        int m = k.dims[1];
        AssertInFastLLM(weights.Count(0) == (uint64_t)bsz * seqlen * heads,
                        "DeepSeekV41IndexerScore error: weights should be [b, s, h].\n");
        std::vector<float> qv, wv, kvv;
        V41ReadFloat(q, qv);
        V41ReadFloat(weights, wv);
        V41ReadFloat(k, kvv);
        output.Allocate();
        float *out = (float*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            for (int t = st; t < end; t++) {
                int b = t / seqlen;
                const float *qrow = qv.data() + (uint64_t)t * heads * dim;
                const float *w = wv.data() + (uint64_t)t * heads;
                float *orow = out + (uint64_t)t * m;
                for (int j = 0; j < m; j++) {
                    const float *krow = kvv.data() + ((uint64_t)b * m + j) * dim;
                    float total = 0.0f;
                    for (int h = 0; h < heads; h++) {
                        const float *qh = qrow + (uint64_t)h * dim;
                        float dot = 0.0f;
                        for (int d = 0; d < dim; d++) {
                            dot += qh[d] * krow[d];
                        }
                        total += std::max(dot, 0.0f) * w[h];
                    }
                    orow[j] = total;
                }
            }
        });
    }

    // ---------------- CandidateBlocks ----------------

    void CpuDeepSeekV41CandidateBlocksOp::Reshape(const std::string &opType, const DataDict &datas,
                                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int blockSize = V41Int(intParams, "blockSize", 8);
        AssertInFastLLM(score.dims.size() == 3 && blockSize > 0,
                        "DeepSeekV41CandidateBlocks error: score should be [b, s, m].\n");
        int numBlocks = (score.dims[2] + blockSize - 1) / blockSize;
        output.dataType = DataType::INT8;
        output.Resize({score.dims[0], score.dims[1], numBlocks});
    }

    void CpuDeepSeekV41CandidateBlocksOp::Run(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int blockSize = V41Int(intParams, "blockSize", 8);
        int topkBlocks = V41Int(intParams, "topkBlocks", 0);
        int compressRatio = V41Int(intParams, "compressRatio", 1);
        int startPos = V41Int(intParams, "startPos", 0);
        int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
        int numBlocks = (m + blockSize - 1) / blockSize;
        AssertInFastLLM(score.dataType == DataType::FLOAT32, "DeepSeekV41CandidateBlocks error: score should be float32.\n");
        output.Allocate();
        const float *sc = (const float*)score.cpuData;
        uint8_t *out = (uint8_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> blockScore(numBlocks);
            std::vector<int> order(numBlocks);
            for (int t = st; t < end; t++) {
                int i = t % seqlen;
                int visible = std::min(m, (startPos + i + 1) / compressRatio);
                const float *row = sc + (uint64_t)t * m;
                for (int k = 0; k < numBlocks; k++) {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int j = k * blockSize; j < std::min(m, (k + 1) * blockSize); j++) {
                        if (j < visible) {
                            mx = std::max(mx, row[j]);
                        }
                    }
                    blockScore[k] = mx;
                }
                if (visible > 0) {
                    blockScore[(visible - 1) / blockSize] = std::numeric_limits<float>::infinity();
                }
                uint8_t *orow = out + (uint64_t)t * numBlocks;
                memset(orow, 0, numBlocks);
                int keep = std::min(topkBlocks, numBlocks);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                                  [&](int a, int b) {
                                      return blockScore[a] > blockScore[b] || (blockScore[a] == blockScore[b] && a < b);
                                  });
                for (int k = 0; k < keep; k++) {
                    if (blockScore[order[k]] > -std::numeric_limits<float>::infinity()) {
                        orow[order[k]] = 1;
                    }
                }
            }
        });
    }

    // ---------------- IndexerTopK ----------------

    void CpuDeepSeekV41IndexerTopKOp::Reshape(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int topK = V41Int(intParams, "topK", 0);
        AssertInFastLLM(score.dims.size() == 3 && topK > 0, "DeepSeekV41IndexerTopK error: score should be [b, s, m].\n");
        output.dataType = DataType::INT32;
        output.Resize({score.dims[0], score.dims[1], std::min(topK, score.dims[2])});
    }

    void CpuDeepSeekV41IndexerTopKOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data *candidates = V41Optional(datas, "candidates");
        Data &output = *(datas.find("output")->second);
        int topK = V41Int(intParams, "topK", 0);
        int compressRatio = V41Int(intParams, "compressRatio", 1);
        int startPos = V41Int(intParams, "startPos", 0);
        int blockSize = V41Int(intParams, "blockSize", 8);
        int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
        int width = std::min(topK, m);
        AssertInFastLLM(score.dataType == DataType::FLOAT32, "DeepSeekV41IndexerTopK error: score should be float32.\n");
        int numBlocks = candidates == nullptr ? 0 : candidates->dims.back();
        output.Allocate();
        const float *sc = (const float*)score.cpuData;
        int32_t *out = (int32_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<int> order;
            std::vector<float> masked(m);
            for (int t = st; t < end; t++) {
                int i = t % seqlen;
                int visible = std::min(m, (startPos + i + 1) / compressRatio);
                const float *row = sc + (uint64_t)t * m;
                const uint8_t *cand = candidates == nullptr ? nullptr :
                    (const uint8_t*)candidates->cpuData + (uint64_t)t * numBlocks;
                order.clear();
                for (int j = 0; j < visible; j++) {
                    if (cand != nullptr && (j / blockSize >= numBlocks || cand[j / blockSize] == 0)) {
                        continue;
                    }
                    order.push_back(j);
                }
                int keep = std::min(width, (int)order.size());
                std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                                  [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
                std::sort(order.begin(), order.begin() + keep);
                int32_t *orow = out + (uint64_t)t * width;
                for (int k = 0; k < width; k++) {
                    orow[k] = k < keep ? order[k] : -1;
                }
            }
        });
    }

    // ---------------- SparseAttention ----------------

    void CpuDeepSeekV41SparseAttentionOp::Reshape(const std::string &opType, const DataDict &datas,
                                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(q.dims.size() == 4, "DeepSeekV41SparseAttention error: q should be [b, s, h, d].\n");
        output.dataType = DataType::BFLOAT16;
        output.Resize(q.dims);
    }

    void CpuDeepSeekV41SparseAttentionOp::Run(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &chunkKV = *(datas.find("chunkKV")->second);
        Data *ringKV = V41Optional(datas, "ringKV");
        Data *compressedKV = V41Optional(datas, "compressedKV");
        Data *cmpIdx = V41Optional(datas, "cmpIdx");
        Data &attnSink = *(datas.find("attnSink")->second);
        Data &output = *(datas.find("output")->second);
        int windowSize = V41Int(intParams, "windowSize", 128);
        int startPos = V41Int(intParams, "startPos", 0);
        float softmaxScale = V41Float(floatParams, "softmaxScale", 1.0f);

        int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
        AssertInFastLLM(chunkKV.dims.size() == 3 && chunkKV.dims[0] == bsz && chunkKV.dims[1] == seqlen &&
                        chunkKV.dims[2] == dim && attnSink.Count(0) >= (uint64_t)heads,
                        "DeepSeekV41SparseAttention error: chunkKV / sink shape mismatch.\n");
        bool hasRing = ringKV != nullptr && ringKV->dims.size() == 3 && ringKV->Count(0) > 0;
        bool hasCompressed = compressedKV != nullptr && cmpIdx != nullptr && compressedKV->Count(0) > 0 &&
                             cmpIdx->Count(0) > 0;
        int ringRows = hasRing ? ringKV->dims[1] : 0;
        int cap = hasCompressed ? compressedKV->dims[1] : 0;
        int topWidth = hasCompressed ? cmpIdx->dims[2] : 0;
        AssertInFastLLM(!hasRing || (ringKV->dims[0] == bsz && ringKV->dims[2] == dim && ringRows == windowSize),
                        "DeepSeekV41SparseAttention error: ring shape mismatch.\n");
        AssertInFastLLM(!hasCompressed || (cmpIdx->dataType == DataType::INT32 && cmpIdx->dims.size() == 3 &&
                                           cmpIdx->dims[0] == bsz && cmpIdx->dims[1] == seqlen &&
                                           compressedKV->dims[2] == dim),
                        "DeepSeekV41SparseAttention error: compressed shape mismatch.\n");
        std::vector<float> qv, chunk, ring, comp, sink;
        V41ReadFloat(q, qv);
        V41ReadFloat(chunkKV, chunk);
        if (hasRing) {
            V41ReadFloat(*ringKV, ring);
        }
        if (hasCompressed) {
            V41ReadFloat(*compressedKV, comp);
        }
        V41ReadFloat(attnSink, sink);
        output.Allocate();
        uint16_t *out = (uint16_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<const float*> rows;
            std::vector<float> scores;
            std::vector<float> acc(dim);
            for (int t = st; t < end; t++) {
                int b = t / seqlen, i = t % seqlen;
                int pos = startPos + i;
                rows.clear();
                for (int p = std::max(0, pos - windowSize + 1); p <= pos; p++) {
                    if (p >= startPos) {
                        rows.push_back(chunk.data() + ((uint64_t)b * seqlen + (p - startPos)) * dim);
                    } else {
                        AssertInFastLLM(hasRing, "DeepSeekV41SparseAttention error: ring cache is missing.\n");
                        rows.push_back(ring.data() + ((uint64_t)b * ringRows + (p % windowSize)) * dim);
                    }
                }
                if (hasCompressed) {
                    const int32_t *idx = (const int32_t*)cmpIdx->cpuData + (uint64_t)t * topWidth;
                    for (int k = 0; k < topWidth; k++) {
                        if (idx[k] >= 0 && idx[k] < cap) {
                            rows.push_back(comp.data() + ((uint64_t)b * cap + idx[k]) * dim);
                        }
                    }
                }
                scores.resize(rows.size());
                for (int h = 0; h < heads; h++) {
                    const float *qrow = qv.data() + ((uint64_t)t * heads + h) * dim;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (size_t k = 0; k < rows.size(); k++) {
                        double dot = 0.0;
                        for (int d = 0; d < dim; d++) {
                            dot += (double)qrow[d] * rows[k][d];
                        }
                        scores[k] = (float)dot * softmaxScale;
                        mx = std::max(mx, scores[k]);
                    }
                    float safeMx = std::isfinite(mx) ? mx : 0.0f;
                    double denom = std::exp((double)sink[h] - safeMx);
                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (size_t k = 0; k < rows.size(); k++) {
                        double w = std::exp((double)scores[k] - safeMx);
                        denom += w;
                        for (int d = 0; d < dim; d++) {
                            acc[d] += (float)w * rows[k][d];
                        }
                    }
                    float inv = (float)(1.0 / std::max(denom, 1e-30));
                    uint16_t *orow = out + ((uint64_t)t * heads + h) * dim;
                    for (int d = 0; d < dim; d++) {
                        orow[d] = Float32ToBFloat16RNEBits(acc[d] * inv);
                    }
                }
            }
        });
    }

    // ---------------- WindowStore ----------------

    void CpuDeepSeekV41WindowStoreOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &chunk = *(datas.find("chunk")->second);
        Data &ring = *(datas.find("ring")->second);
        int startPos = V41Int(intParams, "startPos", 0);
        int windowSize = V41Int(intParams, "windowSize", 128);
        AssertInFastLLM(chunk.dims.size() == 3, "DeepSeekV41WindowStore error: chunk should be [b, s, d].\n");
        int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
        if (ring.dims.size() != 3 || ring.dims[0] != bsz || ring.dims[1] != windowSize || ring.dims[2] != dim ||
            ring.cpuData == nullptr) {
            ring.dataType = chunk.dataType;
            ring.Resize({bsz, windowSize, dim});
            ring.Allocate();
        }
        AssertInFastLLM(ring.dataType == chunk.dataType, "DeepSeekV41WindowStore error: dtype mismatch.\n");
        int unit = chunk.unitSize;
        for (int b = 0; b < bsz; b++) {
            for (int i = std::max(0, seqlen - windowSize); i < seqlen; i++) {
                int slot = (startPos + i) % windowSize;
                memcpy(ring.cpuData + ((uint64_t)b * windowSize + slot) * dim * unit,
                       chunk.cpuData + ((uint64_t)b * seqlen + i) * dim * unit,
                       (uint64_t)dim * unit);
            }
        }
    }
}
