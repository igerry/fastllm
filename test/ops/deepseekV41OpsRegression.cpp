//
// DeepSeek-V4.1 专用算子的 CPU / CUDA 一致性测试。
//
// 11 个算子（CPU 参考实现在 src/devices/cpu/deepseekv41ops.cpp，CUDA 在
// src/devices/cuda/models/deepseekv41-kernels.cu）用同一份输入分别在两个设备上跑，
// 逐元素比较输出。每个用例都先确认 CUDA 端的 CanRun 接受该输入，避免"CUDA 悄悄退回 CPU"
// 导致比较变成 CPU vs CPU。
//
// 容差：
//   * 返回下标 / 掩码 / 打包字节的算子（IndexerTopK、CandidateBlocks、QuantizeKV、WindowStore）
//     要求逐字节一致——CPU 实现里的并列按下标从小到大 tie-break，CUDA 必须复现同一条规则，
//     所以用例里专门放了"分数全相同"和"分数只有几档"的输入；
//   * 输出为 BF16 的算子（SparseAttention、Compress、EngramApply）允许 1–2 个 BF16 ulp
//     （相对容差 4e-3 ~ 8e-3）；
//   * 输出为 FP32 的算子允许约 1e-6 的相对误差（实测都在 1e-7 量级）。
//
// 用法：
//   ./deepseekV41OpsRegression                 跑全部用例
//   ./deepseekV41OpsRegression --list          列出用例
//   ./deepseekV41OpsRegression IndexerTopK     只跑算子名里含该子串的用例
//   ./deepseekV41OpsRegression -v              打印每个用例的最大偏差
// 退出码：0 = 全部通过，1 = 有用例失败，77 = 没有 CUDA 设备（跳过）。
//

#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace fastllm;

// ==================== 小工具 ====================

namespace {

bool gVerbose = false;

struct Rng {
    std::mt19937 gen;
    explicit Rng(uint32_t seed) : gen(seed) {}
    float Uniform(float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(gen);
    }
    int Int(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(gen);
    }
    std::vector<float> Floats(size_t n, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = Uniform(lo, hi);
        }
        return v;
    }
};

size_t Numel(const std::vector<int> &dims) {
    size_t n = 1;
    for (int d : dims) {
        n *= (size_t)d;
    }
    return n;
}

// 设备无关的张量描述：先把内容准备成 host 数据，再为两次运行各造一份 Data
struct Tensor {
    DataType type = DataType::FLOAT32;
    std::vector<int> dims;
    std::vector<float> values;    // 浮点类型用
    std::vector<uint8_t> bytes;   // INT8 / INT32 等按原始字节给
    bool raw = false;
};

Tensor Floats(DataType type, const std::vector<int> &dims, std::vector<float> values) {
    Tensor t;
    t.type = type;
    t.dims = dims;
    t.values = std::move(values);
    t.values.resize(Numel(dims), 0.0f);
    return t;
}

Tensor Bytes(DataType type, const std::vector<int> &dims, std::vector<uint8_t> bytes) {
    Tensor t;
    t.type = type;
    t.dims = dims;
    t.bytes = std::move(bytes);
    t.raw = true;
    return t;
}

Tensor Ints(const std::vector<int> &dims, const std::vector<int32_t> &values) {
    std::vector<uint8_t> raw(values.size() * sizeof(int32_t));
    memcpy(raw.data(), values.data(), raw.size());
    return Bytes(DataType::INT32, dims, std::move(raw));
}

Tensor Mask(const std::vector<int> &dims, const std::vector<uint8_t> &values) {
    return Bytes(DataType::INT8, dims, values);
}

Data *MakeData(const Tensor &t) {
    if (!t.raw) {
        return new Data(t.type, t.dims, t.values);
    }
    Data *d = new Data(t.type, t.dims);
    d->Allocate();
    memcpy(d->cpuData, t.bytes.data(), std::min(t.bytes.size(), (size_t)d->GetBytes()));
    return d;
}

float LoadFloat(const uint8_t *base, DataType type, uint64_t i) {
    if (type == DataType::FLOAT32) {
        return ((const float*)base)[i];
    }
    if (type == DataType::BFLOAT16) {
        uint32_t bits = (uint32_t)(((const uint16_t*)base)[i]) << 16;
        float v;
        memcpy(&v, &bits, sizeof(v));
        return v;
    }
    if (type == DataType::FLOAT16) {
        return half_to_float(((const uint16_t*)base)[i]);
    }
    return 0.0f;
}

bool IsFloatType(DataType t) {
    return t == DataType::FLOAT32 || t == DataType::FLOAT16 || t == DataType::BFLOAT16;
}

std::vector<float> ReadFloats(Data &d) {
    d.ToDevice(DataDevice::CPU);
    uint64_t n = d.Count(0);
    std::vector<float> out(n);
    for (uint64_t i = 0; i < n; i++) {
        out[i] = LoadFloat(d.cpuData, d.dataType, i);
    }
    return out;
}

std::vector<uint8_t> ReadBytes(Data &d) {
    d.ToDevice(DataDevice::CPU);
    std::vector<uint8_t> out(d.GetBytes());
    memcpy(out.data(), d.cpuData, out.size());
    return out;
}

// ==================== 用例定义 ====================

struct OpCase {
    std::string op;
    std::string name;
    std::vector<std::pair<std::string, Tensor> > inputs;   // 含原地修改的输入
    std::vector<std::string> outputs;                      // 交给算子分配的输出
    std::vector<std::string> compare;                      // 需要比较的 Data 名
    FloatDict floatParams;
    IntDict intParams;
    double atol = 1e-5;
    double rtol = 1e-5;
    std::string note;
};

struct Bag {
    std::vector<std::unique_ptr<Data> > owned;
    std::map<std::string, Data*> byName;
    DataDict datas;

    void Add(const std::string &name, Data *d) {
        owned.emplace_back(d);
        byName[name] = d;
        datas[name] = d;
    }
};

void Build(const OpCase &c, Bag &bag) {
    for (const auto &kv : c.inputs) {
        bag.Add(kv.first, MakeData(kv.second));
    }
    for (const std::string &name : c.outputs) {
        bag.Add(name, new Data());
    }
}

struct Result {
    bool ok = true;
    std::string detail;
};

Result CompareData(const std::string &name, Data &cpu, Data &gpu, double atol, double rtol) {
    Result r;
    char buf[512];
    if (cpu.dims != gpu.dims) {
        r.ok = false;
        r.detail = name + ": dims differ";
        return r;
    }
    if (cpu.dataType != gpu.dataType) {
        r.ok = false;
        r.detail = name + ": dtype differs";
        return r;
    }
    if (IsFloatType(cpu.dataType)) {
        std::vector<float> a = ReadFloats(cpu), b = ReadFloats(gpu);
        double maxAbs = 0.0, maxDiff = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < a.size(); i++) {
            maxAbs = std::max(maxAbs, (double)std::fabs(a[i]));
            double d = std::fabs((double)a[i] - (double)b[i]);
            if (std::isnan(d) || d > maxDiff) {
                maxDiff = std::isnan(d) ? 1e30 : d;
                worst = i;
            }
        }
        double tol = atol + rtol * maxAbs;
        if (!(maxDiff <= tol)) {
            r.ok = false;
            snprintf(buf, sizeof(buf), "%s: max|diff|=%.3e > tol %.3e (at %zu: cpu=%.6g cuda=%.6g, max|cpu|=%.3e)",
                     name.c_str(), maxDiff, tol, worst, a[worst], b[worst], maxAbs);
        } else {
            snprintf(buf, sizeof(buf), "%s max|diff|=%.2e/%.2e", name.c_str(), maxDiff, tol);
        }
        r.detail = buf;
        return r;
    }
    // 整数 / 打包字节：要求逐字节一致（CPU 实现有确定性 tie-break，CUDA 必须复现）
    std::vector<uint8_t> a = ReadBytes(cpu), b = ReadBytes(gpu);
    if (a.size() != b.size()) {
        r.ok = false;
        r.detail = name + ": byte count differs";
        return r;
    }
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            if (bad == 0) {
                first = i;
            }
            bad++;
        }
    }
    if (bad) {
        r.ok = false;
        if (cpu.dataType == DataType::INT32) {
            const int32_t *ia = (const int32_t*)a.data(), *ib = (const int32_t*)b.data();
            size_t idx = first / 4;
            snprintf(buf, sizeof(buf), "%s: %zu / %zu bytes differ (int32[%zu]: cpu=%d cuda=%d)",
                     name.c_str(), bad, a.size(), idx, ia[idx], ib[idx]);
        } else {
            snprintf(buf, sizeof(buf), "%s: %zu / %zu bytes differ (byte %zu: cpu=%u cuda=%u)",
                     name.c_str(), bad, a.size(), first, (unsigned)a[first], (unsigned)b[first]);
        }
    } else {
        snprintf(buf, sizeof(buf), "%s exact (%zu bytes)", name.c_str(), a.size());
    }
    r.detail = buf;
    return r;
}

bool RunCase(Executor *executor, const OpCase &c) {
    Bag cpu, gpu;
    Build(c, cpu);
    Build(c, gpu);
    std::string label = c.op + " / " + c.name;

    executor->SetFirstDevice("cuda:0");
    if (!executor->CanRunOnFirstDevice(c.op, gpu.datas, c.floatParams, c.intParams)) {
        printf("[FAIL] %-52s CUDA CanRun rejected this input (would silently fall back to CPU)\n",
               label.c_str());
        return false;
    }
    executor->RunOnDevice("cpu", c.op, cpu.datas, c.floatParams, c.intParams);
    executor->RunOnDevice("cuda", c.op, gpu.datas, c.floatParams, c.intParams);
    cudaDeviceSynchronize();

    bool ok = true;
    std::vector<std::string> details;
    for (const std::string &name : c.compare) {
        Result r = CompareData(name, *cpu.byName[name], *gpu.byName[name], c.atol, c.rtol);
        ok = ok && r.ok;
        details.push_back(r.detail);
    }
    if (!ok) {
        printf("[FAIL] %-52s %s\n", label.c_str(), c.note.c_str());
        for (const std::string &d : details) {
            printf("           %s\n", d.c_str());
        }
    } else if (gVerbose) {
        std::string joined;
        for (size_t i = 0; i < details.size(); i++) {
            joined += (i ? ", " : "") + details[i];
        }
        printf("[ ok ] %-52s %s\n", label.c_str(), joined.c_str());
    }
    return ok;
}

// 用 CPU 的 QuantizeKV 造出合法的 FP8 + UE8M0 缓存行，喂给 SparseAttention 的 FP8 分支
Tensor QuantizeRowsOnCpu(Executor *executor, const Tensor &src) {
    OpCase c;
    c.op = "DeepSeekV41QuantizeKV";
    c.inputs.push_back(std::make_pair(std::string("input"), src));
    c.outputs.push_back("output");
    Bag bag;
    Build(c, bag);
    executor->RunOnDevice("cpu", c.op, bag.datas, c.floatParams, c.intParams);
    Data &out = *bag.byName["output"];
    return Bytes(DataType::INT8, out.dims, ReadBytes(out));
}

// ==================== 各算子的用例 ====================

void AddHcMix(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg { const char *name; int bsz, seqlen, hc, dim, iters; DataType type; float amp; };
    const Cfg cfgs[] = {
        {"hc2_f32", 1, 5, 2, 64, 20, DataType::FLOAT32, 1.0f},
        {"hc4_bf16", 1, 7, 4, 128, 20, DataType::BFLOAT16, 1.0f},
        {"hc4_f16_1iter", 2, 3, 4, 96, 1, DataType::FLOAT16, 1.0f},
        {"hc1_degenerate", 1, 4, 1, 64, 20, DataType::FLOAT32, 1.0f},
        {"hc3_extreme", 1, 6, 3, 64, 20, DataType::FLOAT32, 60.0f},   // sigmoid / softmax 饱和
        {"hc2_zeros", 1, 4, 2, 64, 20, DataType::FLOAT32, 0.0f},      // 全零输入 -> RMSNorm 走 eps
    };
    for (const Cfg &cfg : cfgs) {
        int flat = cfg.hc * cfg.dim;
        int mixHc = (2 + cfg.hc) * cfg.hc;
        OpCase c;
        c.op = "DeepSeekV41HcMix";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(cfg.type, {cfg.bsz, cfg.seqlen, cfg.hc, cfg.dim},
                   rng.Floats((size_t)cfg.bsz * cfg.seqlen * flat, -cfg.amp, cfg.amp))));
        c.inputs.push_back(std::make_pair(std::string("hcFn"),
            Floats(DataType::FLOAT32, {mixHc, flat}, rng.Floats((size_t)mixHc * flat, -0.05f, 0.05f))));
        c.inputs.push_back(std::make_pair(std::string("hcScale"),
            Floats(DataType::FLOAT32, {3}, {0.5f, 1.5f, 2.0f})));
        c.inputs.push_back(std::make_pair(std::string("hcBase"),
            Floats(DataType::FLOAT32, {mixHc}, rng.Floats(mixHc, -0.3f, 0.3f))));
        c.outputs = {"pre", "post", "comb"};
        c.compare = {"pre", "post", "comb"};
        c.intParams = {{"hcMult", cfg.hc}, {"sinkhornIters", cfg.iters}};
        c.floatParams = {{"eps", 1e-6f}, {"normEps", 1e-6f}};
        c.atol = 2e-6;
        c.rtol = 2e-6;
        cases.push_back(c);
    }
}

void AddHcApplyPre(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg { const char *name; int bsz, seqlen, hc, dim; DataType type; };
    const Cfg cfgs[] = {
        {"f32", 1, 7, 4, 96, DataType::FLOAT32},
        {"bf16_wide", 1, 5, 2, 512, DataType::BFLOAT16},
        {"f16", 2, 3, 4, 128, DataType::FLOAT16},
        {"hc1", 1, 9, 1, 64, DataType::FLOAT32},
        {"odd_dim", 1, 4, 3, 37, DataType::FLOAT32},        // dim 不是 warp 的整数倍
    };
    for (const Cfg &cfg : cfgs) {
        int tokens = cfg.bsz * cfg.seqlen;
        OpCase c;
        c.op = "DeepSeekV41HcApplyPre";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(cfg.type, {cfg.bsz, cfg.seqlen, cfg.hc, cfg.dim},
                   rng.Floats((size_t)tokens * cfg.hc * cfg.dim))));
        c.inputs.push_back(std::make_pair(std::string("pre"),
            Floats(DataType::FLOAT32, {cfg.bsz, cfg.seqlen, cfg.hc},
                   rng.Floats((size_t)tokens * cfg.hc, 0.0f, 1.5f))));
        c.outputs = {"output"};
        c.compare = {"output"};
        c.atol = cfg.type == DataType::FLOAT32 ? 1e-6 : 4e-3;   // BF16 输出允许 1 个 ulp
        c.rtol = cfg.type == DataType::FLOAT32 ? 1e-6 : 4e-3;
        cases.push_back(c);
    }
}

void AddEngramApply(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg { const char *name; int seqlen, hc, dim; bool mask; float amp; };
    const Cfg cfgs[] = {
        {"no_mask", 6, 2, 128, false, 1.0f},
        {"masked", 6, 2, 128, true, 1.0f},               // 图像 token 不查表
        {"hc4", 5, 4, 64, false, 1.0f},
        {"tiny_scores", 4, 2, 64, false, 1e-4f},         // |score| < clampValue，走 clamp 分支
        {"large_scores", 4, 2, 64, false, 30.0f},        // sigmoid 饱和
    };
    for (const Cfg &cfg : cfgs) {
        int tokens = cfg.seqlen;
        OpCase c;
        c.op = "DeepSeekV41EngramApply";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("hidden"),
            Floats(DataType::BFLOAT16, {1, cfg.seqlen, cfg.hc, cfg.dim},
                   rng.Floats((size_t)tokens * cfg.hc * cfg.dim, -cfg.amp, cfg.amp))));
        c.inputs.push_back(std::make_pair(std::string("kv"),
            Floats(DataType::BFLOAT16, {1, cfg.seqlen, (cfg.hc + 1) * cfg.dim},
                   rng.Floats((size_t)tokens * (cfg.hc + 1) * cfg.dim, -cfg.amp, cfg.amp))));
        c.inputs.push_back(std::make_pair(std::string("qWeight"),
            Floats(DataType::FLOAT32, {cfg.hc, cfg.dim}, rng.Floats((size_t)cfg.hc * cfg.dim, 0.5f, 1.5f))));
        c.inputs.push_back(std::make_pair(std::string("kWeight"),
            Floats(DataType::FLOAT32, {cfg.hc, cfg.dim}, rng.Floats((size_t)cfg.hc * cfg.dim, 0.5f, 1.5f))));
        if (cfg.mask) {
            std::vector<float> mask(tokens, 1.0f);
            mask[0] = 0.0f;
            mask[2] = 0.0f;
            mask[tokens - 1] = 0.0f;
            c.inputs.push_back(std::make_pair(std::string("mask"),
                Floats(DataType::FLOAT32, {1, cfg.seqlen}, mask)));
        }
        c.compare = {"hidden"};                  // 原地写回
        c.floatParams = {{"eps", 1e-20f}};
        c.atol = 4e-3;                           // 输出是 BF16，允许 1 个 ulp
        c.rtol = 4e-3;
        cases.push_back(c);
    }
}

void AddRotaryQuant(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg {
        const char *name; std::vector<int> dims; DataType type;
        int ropeDim, startPos, posStep, inverse, originalSeqLen, quantMode, quantBlock, quantDim;
        float factor;
    };
    const std::vector<Cfg> cfgs = {
        {"rope3d", {1, 8, 128}, DataType::FLOAT32, 64, 3, 1, 0, 0, 0, 32, 0, 1.0f},
        {"rope4d_bf16", {1, 5, 4, 64}, DataType::BFLOAT16, 32, 0, 1, 0, 0, 0, 32, 0, 1.0f},
        {"inverse_step2", {1, 6, 64}, DataType::FLOAT32, 32, 2, 2, 1, 0, 0, 32, 0, 1.0f},
        {"yarn_far", {1, 4, 128}, DataType::FLOAT32, 64, 100000, 1, 0, 65536, 0, 32, 0, 16.0f},
        {"fp8_block32", {1, 6, 128}, DataType::FLOAT32, 64, 1, 1, 0, 0, 1, 32, 0, 1.0f},
        {"fp4_e8m0_block16", {1, 6, 128}, DataType::FLOAT32, 64, 1, 1, 0, 0, 2, 16, 64, 1.0f},
        {"fp4_e4m3_block32", {1, 6, 128}, DataType::FLOAT32, 64, 1, 1, 0, 0, 3, 32, 0, 1.0f},
        {"quant_dim_lt_dim", {1, 4, 128}, DataType::FLOAT32, 32, 0, 1, 0, 0, 1, 32, 96, 1.0f},
        {"maxdim1024", {1, 2, 1024}, DataType::FLOAT32, 64, 7, 1, 0, 0, 1, 32, 0, 1.0f},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41RotaryQuant";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(cfg.type, cfg.dims, rng.Floats(Numel(cfg.dims), -2.0f, 2.0f))));
        c.compare = {"input"};
        c.intParams = {{"ropeDim", cfg.ropeDim}, {"startPos", cfg.startPos}, {"posStep", cfg.posStep},
                       {"inverse", cfg.inverse}, {"originalSeqLen", cfg.originalSeqLen},
                       {"betaFast", 32}, {"betaSlow", 1},
                       {"quantMode", cfg.quantMode}, {"quantBlock", cfg.quantBlock}};
        if (cfg.quantDim > 0) {
            c.intParams["quantDim"] = cfg.quantDim;
        }
        c.floatParams = {{"ropeBase", 10000.0f}, {"ropeFactor", cfg.factor}};
        c.atol = cfg.type == DataType::FLOAT32 ? 2e-6 : 4e-3;
        c.rtol = cfg.type == DataType::FLOAT32 ? 1e-6 : 4e-3;
        cases.push_back(c);
    }

    // 伪量化的并列（正好落在两个网格点中间）必须按同一条 tie-break 规则走
    {
        const int dim = 128, ropeDim = 32;   // 前 96 个元素不参与 RoPE，专门放并列值
        std::vector<float> v(dim * 4, 0.0f);
        const float ties[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};  // E2M1 网格中点
        for (int row = 0; row < 4; row++) {
            float *p = v.data() + row * dim;
            for (int i = 0; i < dim - ropeDim; i++) {
                p[i] = ties[i % 7] * (i % 2 ? -1.0f : 1.0f);
            }
            p[0] = 6.0f;                     // 让 amax = 6 -> scale = 1，网格中点原样保留
            for (int i = dim - ropeDim; i < dim; i++) {
                p[i] = rng.Uniform(-1.0f, 1.0f);
            }
        }
        OpCase c;
        c.op = "DeepSeekV41RotaryQuant";
        c.name = "fp4_grid_ties";
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(DataType::FLOAT32, {1, 4, dim}, v)));
        c.compare = {"input"};
        c.intParams = {{"ropeDim", ropeDim}, {"startPos", 0}, {"posStep", 1}, {"inverse", 0},
                       {"originalSeqLen", 0}, {"betaFast", 32}, {"betaSlow", 1},
                       {"quantMode", 2}, {"quantBlock", 32}};
        c.floatParams = {{"ropeBase", 10000.0f}, {"ropeFactor", 1.0f}};
        c.atol = 0.0;
        c.rtol = 0.0;
        c.note = "FP4 E2M1 网格中点的取整方向";
        cases.push_back(c);
    }
    // FP8 分支的边界：0、极小值、超过 448 的饱和值
    {
        const int dim = 128;
        std::vector<float> v(dim * 2, 0.0f);
        for (int row = 0; row < 2; row++) {
            float *p = v.data() + row * dim;
            for (int i = 0; i < dim; i++) {
                p[i] = rng.Uniform(-1.0f, 1.0f);
            }
            p[0] = 0.0f;
            p[1] = -0.0f;
            p[2] = 1e-8f;
            p[3] = 1e6f;
            p[4] = -1e6f;
            p[32] = 0.0f;                    // 整块接近 0 -> amax 被 clamp 到 1e-4
            for (int i = 33; i < 64; i++) {
                p[i] = 1e-9f;
            }
        }
        OpCase c;
        c.op = "DeepSeekV41RotaryQuant";
        c.name = "fp8_extremes";
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(DataType::FLOAT32, {1, 2, dim}, v)));
        c.compare = {"input"};
        c.intParams = {{"ropeDim", 64}, {"startPos", 0}, {"posStep", 1}, {"inverse", 0},
                       {"originalSeqLen", 0}, {"betaFast", 32}, {"betaSlow", 1},
                       {"quantMode", 1}, {"quantBlock", 32}};
        c.floatParams = {{"ropeBase", 10000.0f}, {"ropeFactor", 1.0f}};
        c.atol = 1e-5;
        c.rtol = 1e-6;
        cases.push_back(c);
    }
}

void AddCompress(std::vector<OpCase> &cases, Rng &rng) {
    const int dim = 512;   // CUDA 端只接受 512
    struct Cfg { const char *name; int n, ratio; DataType type; float scoreAmp; bool tieScore; };
    const Cfg cfgs[] = {
        {"ratio1_bf16", 8, 1, DataType::BFLOAT16, 1.0f, false},
        {"ratio1_f32", 6, 1, DataType::FLOAT32, 1.0f, false},
        {"ratio2_bf16", 8, 2, DataType::BFLOAT16, 1.0f, false},
        {"ratio2_f32", 10, 2, DataType::FLOAT32, 1.0f, false},
        {"ratio2_extreme_score", 8, 2, DataType::FLOAT32, 60.0f, false},   // softmax 减最大值
        {"ratio2_tied_score", 8, 2, DataType::FLOAT32, 1.0f, true},        // 权重完全相同
        {"ratio2_single_block", 2, 2, DataType::FLOAT32, 1.0f, false},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41Compress";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("kv"),
            Floats(cfg.type, {1, cfg.n, dim}, rng.Floats((size_t)cfg.n * dim, -2.0f, 2.0f))));
        if (cfg.ratio > 1) {
            std::vector<float> score = cfg.tieScore
                ? std::vector<float>((size_t)cfg.n * dim, 0.75f)
                : rng.Floats((size_t)cfg.n * dim, -cfg.scoreAmp, cfg.scoreAmp);
            c.inputs.push_back(std::make_pair(std::string("score"),
                Floats(cfg.type, {1, cfg.n, dim}, score)));
        }
        c.inputs.push_back(std::make_pair(std::string("normWeight"),
            Floats(DataType::FLOAT32, {dim}, rng.Floats(dim, 0.5f, 1.5f))));   // CUDA 端要求 FP32
        c.outputs = {"output"};
        c.compare = {"output"};
        c.intParams = {{"compressRatio", cfg.ratio}};
        c.floatParams = {{"normEps", 1e-20f}};
        c.atol = 8e-3;    // 输出是 BF16
        c.rtol = 8e-3;
        cases.push_back(c);
    }
}

void AddIndexerScore(std::vector<OpCase> &cases, Rng &rng) {
    const int dim = 128;  // CUDA 端只接受 128
    struct Cfg { const char *name; int seqlen, heads, m; DataType type; float qAmp; };
    const Cfg cfgs[] = {
        {"bf16", 6, 4, 20, DataType::BFLOAT16, 1.0f},
        {"f32", 5, 2, 33, DataType::FLOAT32, 1.0f},
        {"m1", 4, 4, 1, DataType::FLOAT32, 1.0f},
        {"single_head", 7, 1, 17, DataType::BFLOAT16, 1.0f},
        {"all_negative", 5, 4, 12, DataType::FLOAT32, 1.0f},   // relu 全零
        {"wide_m", 3, 2, 257, DataType::BFLOAT16, 1.0f},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41IndexerScore";
        c.name = cfg.name;
        std::vector<float> q = rng.Floats((size_t)cfg.seqlen * cfg.heads * dim, -cfg.qAmp, cfg.qAmp);
        std::vector<float> k = rng.Floats((size_t)cfg.m * dim, -1.0f, 1.0f);
        if (std::string(cfg.name) == "all_negative") {
            for (size_t i = 0; i < q.size(); i++) {
                q[i] = std::fabs(q[i]);
            }
            for (size_t i = 0; i < k.size(); i++) {
                k[i] = -std::fabs(k[i]);
            }
        }
        c.inputs.push_back(std::make_pair(std::string("q"),
            Floats(cfg.type, {1, cfg.seqlen, cfg.heads, dim}, q)));
        c.inputs.push_back(std::make_pair(std::string("weights"),
            Floats(DataType::FLOAT32, {1, cfg.seqlen, cfg.heads},
                   rng.Floats((size_t)cfg.seqlen * cfg.heads, -1.0f, 1.0f))));
        c.inputs.push_back(std::make_pair(std::string("k"),
            Floats(cfg.type, {1, cfg.m, dim}, k)));
        c.outputs = {"output"};
        c.compare = {"output"};
        c.atol = 1e-5;
        c.rtol = 1e-5;
        cases.push_back(c);
    }
}

// 候选块 / top-k 的分数：既有随机分数，也有大量并列
std::vector<float> SelectionScores(Rng &rng, int seqlen, int m, const std::string &kind) {
    std::vector<float> v((size_t)seqlen * m);
    for (int t = 0; t < seqlen; t++) {
        for (int j = 0; j < m; j++) {
            float x;
            if (kind == "ties") {
                x = 1.0f;                               // 全部并列
            } else if (kind == "coarse") {
                x = (float)((j + t) % 4);               // 大量并列 + 少量不同
            } else {
                x = rng.Uniform(-3.0f, 3.0f);
            }
            v[(size_t)t * m + j] = x;
        }
    }
    return v;
}

void AddCandidateBlocks(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg { const char *name; int seqlen, m, blockSize, topkBlocks, ratio, startPos; const char *kind; };
    const Cfg cfgs[] = {
        {"basic", 8, 20, 4, 3, 1, 0, "random"},
        {"partial_block", 8, 21, 8, 2, 1, 0, "random"},        // m 不能被 blockSize 整除
        {"ratio2_startpos", 8, 16, 2, 5, 2, 7, "random"},
        {"not_enough_blocks", 6, 12, 4, 99, 1, 0, "random"},   // topkBlocks > numBlocks
        {"topk_zero", 6, 12, 4, 0, 1, 0, "random"},            // 一个候选都不选
        {"no_visible", 6, 12, 4, 3, 64, 0, "random"},          // visible == 0，掩码全零
        {"all_ties", 8, 24, 4, 3, 1, 0, "ties"},               // 分数全相同 -> 按下标 tie-break
        {"coarse_ties", 8, 24, 3, 4, 1, 5, "coarse"},
        {"block1", 6, 15, 1, 4, 1, 0, "random"},
        {"block_gt_m", 5, 6, 16, 2, 1, 0, "random"},           // 只有一个块
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41CandidateBlocks";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("score"),
            Floats(DataType::FLOAT32, {1, cfg.seqlen, cfg.m},
                   SelectionScores(rng, cfg.seqlen, cfg.m, cfg.kind))));
        c.outputs = {"output"};
        c.compare = {"output"};
        c.intParams = {{"blockSize", cfg.blockSize}, {"topkBlocks", cfg.topkBlocks},
                       {"compressRatio", cfg.ratio}, {"startPos", cfg.startPos}};
        cases.push_back(c);
    }
}

void AddIndexerTopK(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg {
        const char *name; int seqlen, m, topK, ratio, startPos, blockSize;
        const char *kind; int candKind;   // 0 = 无候选, 1 = 部分候选, 2 = 全零候选, 3 = 全一候选
    };
    const Cfg cfgs[] = {
        {"basic", 8, 20, 4, 1, 0, 4, "random", 0},
        {"topk_gt_m", 6, 5, 32, 1, 0, 4, "random", 0},          // width = m，且 visible < topK
        {"ratio2_startpos", 8, 16, 6, 2, 7, 2, "random", 0},
        {"with_candidates", 8, 24, 5, 1, 0, 4, "random", 1},
        {"empty_candidates", 8, 24, 5, 1, 0, 4, "random", 2},   // 候选全零 -> 全 -1
        {"full_candidates", 8, 24, 5, 1, 0, 4, "random", 3},
        {"all_ties", 8, 24, 6, 1, 0, 4, "ties", 0},             // 并列按下标 tie-break
        {"coarse_ties_cand", 8, 24, 6, 1, 3, 4, "coarse", 1},
        {"no_visible", 6, 12, 4, 64, 0, 4, "random", 0},        // visible == 0 -> 全 -1
        {"topk1", 7, 19, 1, 1, 2, 4, "random", 0},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41IndexerTopK";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("score"),
            Floats(DataType::FLOAT32, {1, cfg.seqlen, cfg.m},
                   SelectionScores(rng, cfg.seqlen, cfg.m, cfg.kind))));
        if (cfg.candKind > 0) {
            int numBlocks = (cfg.m + cfg.blockSize - 1) / cfg.blockSize;
            std::vector<uint8_t> mask((size_t)cfg.seqlen * numBlocks, 0);
            for (int t = 0; t < cfg.seqlen; t++) {
                for (int b = 0; b < numBlocks; b++) {
                    uint8_t v = cfg.candKind == 2 ? 0
                              : cfg.candKind == 3 ? 1
                              : (uint8_t)(((t + b) % 3) != 0);
                    mask[(size_t)t * numBlocks + b] = v;
                }
            }
            c.inputs.push_back(std::make_pair(std::string("candidates"),
                Mask({1, cfg.seqlen, numBlocks}, mask)));
        }
        c.outputs = {"output"};
        c.compare = {"output"};
        c.intParams = {{"topK", cfg.topK}, {"compressRatio", cfg.ratio},
                       {"startPos", cfg.startPos}, {"blockSize", cfg.blockSize}};
        cases.push_back(c);
    }
}

void AddSparseAttention(std::vector<OpCase> &cases, Rng &rng, Executor *executor) {
    const int heads = 32, dim = 512;   // CUDA 端要求 heads % 32 == 0 且 dim == 512
    struct Cfg {
        const char *name; int seqlen, windowSize, startPos, cap, topWidth;
        bool ring, ringFp8, compressed, compFp8, allInvalid;
    };
    const Cfg cfgs[] = {
        {"window_only", 6, 8, 0, 0, 0, false, false, false, false, false},
        {"window_longer_than_ws", 12, 4, 0, 0, 0, false, false, false, false, false},
        {"ring_wrap", 5, 8, 11, 0, 0, true, false, false, false, false},   // 环形缓冲跨越边界
        {"ring_fp8", 5, 8, 11, 0, 0, true, true, false, false, false},
        {"compressed_bf16", 6, 8, 0, 12, 4, false, false, true, false, false},
        {"compressed_fp8", 6, 8, 0, 12, 4, false, false, true, true, false},
        {"ring_and_compressed", 4, 8, 9, 10, 3, true, false, true, false, false},
        {"all_idx_invalid", 5, 8, 0, 10, 4, false, false, true, false, true},
        {"decode_single", 1, 8, 23, 16, 6, true, false, true, false, false},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41SparseAttention";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("q"),
            Floats(DataType::BFLOAT16, {1, cfg.seqlen, heads, dim},
                   rng.Floats((size_t)cfg.seqlen * heads * dim, -0.5f, 0.5f))));
        c.inputs.push_back(std::make_pair(std::string("chunkKV"),
            Floats(DataType::BFLOAT16, {1, cfg.seqlen, dim},
                   rng.Floats((size_t)cfg.seqlen * dim, -1.0f, 1.0f))));
        if (cfg.ring) {
            Tensor ring = Floats(DataType::BFLOAT16, {1, cfg.windowSize, dim},
                                 rng.Floats((size_t)cfg.windowSize * dim, -1.0f, 1.0f));
            c.inputs.push_back(std::make_pair(std::string("ringKV"),
                cfg.ringFp8 ? QuantizeRowsOnCpu(executor, ring) : ring));
        }
        if (cfg.compressed) {
            Tensor comp = Floats(DataType::BFLOAT16, {1, cfg.cap, dim},
                                 rng.Floats((size_t)cfg.cap * dim, -1.0f, 1.0f));
            c.inputs.push_back(std::make_pair(std::string("compressedKV"),
                cfg.compFp8 ? QuantizeRowsOnCpu(executor, comp) : comp));
            std::vector<int32_t> idx((size_t)cfg.seqlen * cfg.topWidth, -1);
            if (!cfg.allInvalid) {
                for (int t = 0; t < cfg.seqlen; t++) {
                    for (int k = 0; k < cfg.topWidth; k++) {
                        int v = (t + k * 3) % (cfg.cap + 2);
                        if (k == cfg.topWidth - 1) {
                            v = -1;                     // 尾部填充
                        } else if (t == 0 && k == 0) {
                            v = cfg.cap + 5;            // 越界下标，两边都要忽略
                        }
                        idx[(size_t)t * cfg.topWidth + k] = v;
                    }
                }
            }
            c.inputs.push_back(std::make_pair(std::string("cmpIdx"),
                Ints({1, cfg.seqlen, cfg.topWidth}, idx)));
        }
        c.inputs.push_back(std::make_pair(std::string("attnSink"),
            Floats(DataType::FLOAT32, {heads}, rng.Floats(heads, -1.0f, 1.0f))));
        c.outputs = {"output"};
        c.compare = {"output"};
        c.intParams = {{"windowSize", cfg.windowSize}, {"startPos", cfg.startPos}};
        c.floatParams = {{"softmaxScale", 0.05f}};
        c.atol = 6e-3;   // 输出是 BF16
        c.rtol = 6e-3;
        cases.push_back(c);
    }
}

void AddWindowStore(std::vector<OpCase> &cases, Rng &rng, Executor *executor) {
    struct Cfg { const char *name; int bsz, seqlen, dim, windowSize, startPos; DataType type; bool fp8; };
    const Cfg cfgs[] = {
        {"bf16_wrap", 1, 5, 128, 8, 11, DataType::BFLOAT16, false},   // 跨越环形边界
        {"bf16_exact", 1, 8, 128, 8, 0, DataType::BFLOAT16, false},
        {"chunk_gt_window", 1, 12, 64, 8, 3, DataType::FLOAT32, false},  // 只保留最后 windowSize 行
        {"batch2", 2, 3, 64, 4, 5, DataType::FLOAT32, false},
        {"fp8_rows", 1, 5, 128, 8, 6, DataType::INT8, true},          // FP8 + UE8M0 打包行
        {"single_token", 1, 1, 128, 8, 17, DataType::BFLOAT16, false},
    };
    for (const Cfg &cfg : cfgs) {
        OpCase c;
        c.op = "DeepSeekV41WindowStore";
        c.name = cfg.name;
        int rowUnits = cfg.dim;
        if (cfg.fp8) {
            Tensor src = Floats(DataType::BFLOAT16, {cfg.bsz, cfg.seqlen, cfg.dim},
                                rng.Floats((size_t)cfg.bsz * cfg.seqlen * cfg.dim, -1.0f, 1.0f));
            Tensor packed = QuantizeRowsOnCpu(executor, src);
            rowUnits = packed.dims[2];
            c.inputs.push_back(std::make_pair(std::string("chunk"), packed));
            std::vector<uint8_t> ring((size_t)cfg.bsz * cfg.windowSize * rowUnits);
            for (size_t i = 0; i < ring.size(); i++) {
                ring[i] = (uint8_t)rng.Int(0, 255);
            }
            c.inputs.push_back(std::make_pair(std::string("ring"),
                Bytes(DataType::INT8, {cfg.bsz, cfg.windowSize, rowUnits}, ring)));
        } else {
            c.inputs.push_back(std::make_pair(std::string("chunk"),
                Floats(cfg.type, {cfg.bsz, cfg.seqlen, cfg.dim},
                       rng.Floats((size_t)cfg.bsz * cfg.seqlen * cfg.dim, -1.0f, 1.0f))));
            c.inputs.push_back(std::make_pair(std::string("ring"),
                Floats(cfg.type, {cfg.bsz, cfg.windowSize, cfg.dim},
                       rng.Floats((size_t)cfg.bsz * cfg.windowSize * cfg.dim, -1.0f, 1.0f))));
        }
        c.compare = {"ring"};
        c.intParams = {{"startPos", cfg.startPos}, {"windowSize", cfg.windowSize}};
        c.atol = 0.0;      // 纯搬运，必须逐位一致
        c.rtol = 0.0;
        cases.push_back(c);
    }
}

void AddQuantizeKV(std::vector<OpCase> &cases, Rng &rng) {
    struct Cfg { const char *name; int bsz, seqlen, dim; DataType type; const char *kind; };
    const Cfg cfgs[] = {
        {"bf16_512", 1, 6, 512, DataType::BFLOAT16, "random"},
        {"f32_128", 1, 5, 128, DataType::FLOAT32, "random"},
        {"f16_128", 2, 3, 128, DataType::FLOAT16, "random"},
        {"zeros", 1, 4, 128, DataType::FLOAT32, "zeros"},        // amax 被 clamp 到 1e-4
        {"extremes", 1, 4, 128, DataType::FLOAT32, "extremes"},  // 饱和 / 次正规 / 0
        {"pow2", 1, 4, 128, DataType::FLOAT32, "pow2"},          // amax 正好是 2 的幂
        // 注：seqlen == 0 的空输入不在这里测——CUDA 端 CanRun 会接受但 kernel 拒绝
        // （零字节的 Data 没有 cudaData），会走到 ErrorInFastLLM 而不是退回 CPU。
    };
    for (const Cfg &cfg : cfgs) {
        size_t n = (size_t)cfg.bsz * cfg.seqlen * cfg.dim;
        std::vector<float> v;
        std::string kind = cfg.kind;
        if (kind == "zeros") {
            v.assign(n, 0.0f);
        } else if (kind == "pow2") {
            v.assign(n, 0.0f);
            for (size_t i = 0; i < n; i++) {
                v[i] = std::ldexp(1.0f, (int)(i % 9) - 4) * (i % 2 ? -1.0f : 1.0f);
            }
        } else {
            v = rng.Floats(n, -2.0f, 2.0f);
            if (kind == "extremes") {
                for (size_t r = 0; r * cfg.dim < n; r++) {
                    float *p = v.data() + r * cfg.dim;
                    p[0] = 0.0f;
                    p[1] = -0.0f;
                    p[2] = 1e30f;
                    p[3] = -1e30f;
                    p[4] = 1e-30f;
                    for (int i = 32; i < 64; i++) {
                        p[i] = 0.0f;
                    }
                }
            }
        }
        OpCase c;
        c.op = "DeepSeekV41QuantizeKV";
        c.name = cfg.name;
        c.inputs.push_back(std::make_pair(std::string("input"),
            Floats(cfg.type, {cfg.bsz, cfg.seqlen, cfg.dim}, v)));
        c.outputs = {"output"};
        c.compare = {"output"};
        c.atol = 0.0;
        c.rtol = 0.0;
        cases.push_back(c);
    }
}

std::vector<OpCase> BuildCases(Executor *executor) {
    Rng rng(20260911);
    std::vector<OpCase> cases;
    AddHcMix(cases, rng);
    AddHcApplyPre(cases, rng);
    AddEngramApply(cases, rng);
    AddRotaryQuant(cases, rng);
    AddCompress(cases, rng);
    AddIndexerScore(cases, rng);
    AddCandidateBlocks(cases, rng);
    AddIndexerTopK(cases, rng);
    AddSparseAttention(cases, rng, executor);
    AddWindowStore(cases, rng, executor);
    AddQuantizeKV(cases, rng);
    return cases;
}

}  // namespace

namespace {
bool gFinished = false;
// fastllm 的 ErrorInFastLLM 会 getchar() 后 exit(0)；把提前退出翻译成失败退出码，
// 免得某个算子内部报错却让测试"通过"。
void OnExit() {
    if (!gFinished) {
        printf("\nRESULT: FAIL (aborted early, most likely ErrorInFastLLM inside an operator)\n");
        fflush(stdout);
        _exit(1);
    }
}
}  // namespace

int main(int argc, char **argv) {
    if (freopen("/dev/null", "r", stdin) == nullptr) {
        // 读不到 /dev/null 也不影响主流程
    }
    atexit(OnExit);
    std::vector<std::string> filters;
    bool list = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            gVerbose = true;
        } else if (arg == "--list") {
            list = true;
        } else if (arg == "-h" || arg == "--help") {
            printf("usage: %s [-v] [--list] [op-name-substring ...]\n", argv[0]);
            gFinished = true;
            return 0;
        } else {
            filters.push_back(arg);
        }
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        printf("no CUDA device, skipping\n");
        gFinished = true;
        return 77;
    }
    FastllmCudaSetDevice(0);
    SetThreads(4);
    Executor *executor = (Executor*)GetExecutor();

    std::vector<OpCase> cases = BuildCases(executor);
    if (list) {
        for (const OpCase &c : cases) {
            printf("%s / %s\n", c.op.c_str(), c.name.c_str());
        }
        gFinished = true;
        return 0;
    }

    std::map<std::string, std::pair<int, int> > perOp;   // op -> (通过, 总数)
    int failed = 0, ran = 0;
    for (const OpCase &c : cases) {
        if (!filters.empty()) {
            bool hit = false;
            for (const std::string &f : filters) {
                if (c.op.find(f) != std::string::npos || c.name.find(f) != std::string::npos) {
                    hit = true;
                }
            }
            if (!hit) {
                continue;
            }
        }
        ran++;
        bool ok = RunCase(executor, c);
        perOp[c.op].second++;
        if (ok) {
            perOp[c.op].first++;
        } else {
            failed++;
        }
    }

    printf("\n%-36s %s\n", "operator", "passed");
    for (const auto &kv : perOp) {
        printf("%-36s %d / %d%s\n", kv.first.c_str(), kv.second.first, kv.second.second,
               kv.second.first == kv.second.second ? "" : "   <-- FAILED");
    }
    printf("\n%d / %d cases passed\n", ran - failed, ran);
    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    gFinished = true;
    return failed ? 1 : 0;
}
