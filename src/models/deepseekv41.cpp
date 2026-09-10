//
// DeepSeek-V4.1 系列模型实现。架构说明见 include/models/deepseekv41.h。
//
// 本文件实现通用（CUDA / CPU 混合）路径：
//   * 注意力、Hyper-Connections、indexer 等在执行器选择的设备上运行（通常是 GPU）；
//   * 路由专家通过 MergeMOEBlock 交给 MoE 设备（cpu / numa / cuda）；
//   * Engram 哈希表以 FP8 + UE8M0 scale 原样保存在 CPU 内存中（每层约 100GB），
//     查表在 CPU 完成，后续 wkv 投影与门控在 GPU 完成。
//

#include "deepseekv41.h"

#include "baseblock.h"
#include "executor.h"
#include "utils.h"
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fastllm {
    namespace {
        // ---------------- 配置读取 ----------------

        bool V41HasKey(const WeightMap &weight, const std::string &key) {
            return weight.dicts.find(key) != weight.dicts.end();
        }

        int V41DictInt(const WeightMap &weight, const std::string &key, int fallback) {
            auto it = weight.dicts.find(key);
            return it == weight.dicts.end() ? fallback : atoi(it->second.c_str());
        }

        float V41DictFloat(const WeightMap &weight, const std::string &key, float fallback) {
            auto it = weight.dicts.find(key);
            return it == weight.dicts.end() ? fallback : (float)atof(it->second.c_str());
        }

        std::vector<int64_t> V41DictInt64Array(const WeightMap &weight, const std::string &key) {
            std::vector<int64_t> ret;
            auto it = weight.dicts.find(key);
            if (it == weight.dicts.end()) {
                return ret;
            }
            std::string err;
            auto json = json11::Json::parse(it->second, err);
            if (!err.empty() || !json.is_array()) {
                return ret;
            }
            for (auto &item : json.array_items()) {
                ret.push_back((int64_t)item.number_value());
            }
            return ret;
        }

        std::vector<int> V41DictIntArray(const WeightMap &weight, const std::string &key) {
            std::vector<int> ret;
            for (int64_t v : V41DictInt64Array(weight, key)) {
                ret.push_back((int)v);
            }
            return ret;
        }

        bool V41Contains(const std::vector<int> &values, int v) {
            return std::find(values.begin(), values.end(), v) != values.end();
        }

        bool V41StartsWith(const std::string &s, const std::string &prefix) {
            return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
        }

        bool V41EndsWith(const std::string &s, const std::string &suffix) {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool V41EnvFlag(const char *name) {
            const char *v = std::getenv(name);
            return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
        }

        // ---------------- 算子封装 ----------------

        Executor &V41Executor() {
            return *((Executor*)GetExecutor());
        }

        std::vector<int> V41ReadTokenIds(const Data &inputIds) {
            Data cpuIds;
            const Data *src = &inputIds;
            if (inputIds.dataDevice != DataDevice::CPU) {
                cpuIds.CopyFrom(inputIds);
                cpuIds.ToDevice(DataDevice::CPU);
                src = &cpuIds;
            }
            std::vector<int> ret;
            uint64_t n = src->Count(0);
            for (uint64_t i = 0; i < n; i++) {
                if (src->dataType == DataType::FLOAT32) {
                    ret.push_back((int)(((const float*)src->cpuData)[i] + 1e-6));
                } else if (src->dataType == DataType::INT32) {
                    ret.push_back(((const int32_t*)src->cpuData)[i]);
                } else {
                    ErrorInFastLLM("DeepSeekV41: unsupported inputIds dtype.");
                }
            }
            return ret;
        }

        void V41RMSNormBF16(const Data &input, Data &weight, float eps, Data &output) {
            RMSNorm(input, weight, eps, output);
            ToDataType(output, DataType::BFLOAT16);
        }

        void V41HcMix(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                      int hcMult, int sinkhornIters, float eps, float normEps,
                      Data &pre, Data &post, Data &comb) {
            V41Executor().Run("DeepSeekV41HcMix", {
                {"input", (Data*)&input}, {"hcFn", &hcFn}, {"hcScale", &hcScale}, {"hcBase", &hcBase},
                {"pre", &pre}, {"post", &post}, {"comb", &comb}
            }, {{"eps", eps}, {"normEps", normEps}}, {{"hcMult", hcMult}, {"sinkhornIters", sinkhornIters}});
        }

        void V41HcApplyPre(const Data &input, const Data &pre, Data &output) {
            V41Executor().Run("DeepSeekV41HcApplyPre", {
                {"input", (Data*)&input}, {"pre", (Data*)&pre}, {"output", &output}
            }, {}, {});
        }

        void V41EngramApply(Data &hidden, const Data &kv, Data &qWeight, Data &kWeight,
                            const Data *mask, float eps) {
            DataDict datas = {
                {"hidden", &hidden}, {"kv", (Data*)&kv}, {"qWeight", &qWeight}, {"kWeight", &kWeight}
            };
            if (mask != nullptr) {
                datas["mask"] = (Data*)mask;
            }
            V41Executor().Run("DeepSeekV41EngramApply", datas, {{"eps", eps}}, {});
        }

        struct V41RopeParams {
            int ropeDim;
            float base;
            int originalSeqLen;
            float factor;
            int betaFast;
            int betaSlow;
        };

        void V41RotaryQuant(Data &x, const V41RopeParams &rope, int startPos, int posStep,
                            bool inverse, int quantMode, int quantBlock, int quantDim = -1) {
            static const bool disableFakeQuant = V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            if (disableFakeQuant) {
                quantMode = 0;
            }
            IntDict ints = {
                {"ropeDim", rope.ropeDim}, {"startPos", startPos}, {"posStep", posStep},
                {"inverse", inverse ? 1 : 0}, {"originalSeqLen", rope.originalSeqLen},
                {"betaFast", rope.betaFast}, {"betaSlow", rope.betaSlow},
                {"quantMode", quantMode}, {"quantBlock", quantBlock}
            };
            if (quantDim > 0) {
                ints["quantDim"] = quantDim;
            }
            V41Executor().Run("DeepSeekV41RotaryQuant", {{"input", &x}},
                              {{"ropeBase", rope.base}, {"ropeFactor", rope.factor}}, ints);
        }

        void V41Compress(const Data &kv, const Data *score, Data &normWeight, int ratio, float normEps, Data &output) {
            DataDict datas = {{"kv", (Data*)&kv}, {"normWeight", &normWeight}, {"output", &output}};
            if (score != nullptr) {
                datas["score"] = (Data*)score;
            }
            V41Executor().Run("DeepSeekV41Compress", datas, {{"normEps", normEps}}, {{"compressRatio", ratio}});
        }

        void V41IndexerScore(const Data &q, const Data &weights, const Data &k, Data &output) {
            V41Executor().Run("DeepSeekV41IndexerScore", {
                {"q", (Data*)&q}, {"weights", (Data*)&weights}, {"k", (Data*)&k}, {"output", &output}
            }, {}, {});
        }

        void V41CandidateBlocks(const Data &score, int blockSize, int topkBlocks, int ratio, int startPos, Data &output) {
            V41Executor().Run("DeepSeekV41CandidateBlocks", {
                {"score", (Data*)&score}, {"output", &output}
            }, {}, {{"blockSize", blockSize}, {"topkBlocks", topkBlocks},
                    {"compressRatio", ratio}, {"startPos", startPos}});
        }

        void V41IndexerTopK(const Data &score, const Data *candidates, int topK, int ratio, int startPos,
                            int blockSize, Data &output) {
            DataDict datas = {{"score", (Data*)&score}, {"output", &output}};
            if (candidates != nullptr) {
                datas["candidates"] = (Data*)candidates;
            }
            V41Executor().Run("DeepSeekV41IndexerTopK", datas, {},
                              {{"topK", topK}, {"compressRatio", ratio}, {"startPos", startPos},
                               {"blockSize", blockSize}});
        }

        void V41SparseAttention(const Data &q, const Data &chunkKV, const Data *ringKV,
                                const Data *compressedKV, const Data *cmpIdx, Data &attnSink,
                                int windowSize, int startPos, float softmaxScale, Data &output) {
            DataDict datas = {
                {"q", (Data*)&q}, {"chunkKV", (Data*)&chunkKV}, {"attnSink", &attnSink}, {"output", &output}
            };
            if (ringKV != nullptr && ringKV->dims.size() == 3) {
                datas["ringKV"] = (Data*)ringKV;
            }
            if (compressedKV != nullptr && cmpIdx != nullptr && compressedKV->dims.size() == 3 &&
                compressedKV->dims[1] > 0 && cmpIdx->dims.size() == 3) {
                datas["compressedKV"] = (Data*)compressedKV;
                datas["cmpIdx"] = (Data*)cmpIdx;
            }
            V41Executor().Run("DeepSeekV41SparseAttention", datas, {{"softmaxScale", softmaxScale}},
                              {{"windowSize", windowSize}, {"startPos", startPos}});
        }

        void V41WindowStore(const Data &chunkKV, Data &ring, int startPos, int windowSize) {
            V41Executor().Run("DeepSeekV41WindowStore", {
                {"chunk", (Data*)&chunkKV}, {"ring", &ring}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        // 在 axis=1 上追加行（预扩容 + CatDirect），cache 需为 [b, cap, d]
        void V41AppendRows(Data &cache, const Data &rows) {
            const int unitLen = 256;
            if (cache.dims.size() == 0 && cache.expansionDims.size() == 0) {
                cache.dataType = rows.dataType;
                cache.UpdateUnitSize();
            }
            while ((cache.dims.size() == 0 &&
                    (cache.expansionDims.size() == 0 || rows.dims[1] > cache.expansionDims[1])) ||
                   (cache.dims.size() > 0 && cache.dims[1] + rows.dims[1] > cache.expansionDims[1])) {
                std::vector<int> newDims;
                if (cache.Count(0) == 0 || cache.dims.size() == 0) {
                    newDims = {rows.dims[0], ((rows.dims[1] - 1) / unitLen + 1) * unitLen, rows.dims[2]};
                } else {
                    newDims = cache.dims;
                    newDims[1] += std::max(((rows.dims[1] - 1) / unitLen + 1) * unitLen, cache.dims[1] / 2);
                }
                cache.Expansion(newDims);
            }
            CatDirect(cache, rows, 1);
        }

        // 与 V4 相同的占位 KV，用于让通用调度器统计上下文长度
        void V41UpdateStubPastKeyValues(std::vector<std::pair<Data, Data> > &pastKeyValues,
                                        int totalLen, int blocks) {
            if (pastKeyValues.empty()) {
                return;
            }
            // 通用调度器用 expansionDims 判断请求是否已激活，并在 decode 时读取
            // expansionDims[1]；预留容量必须严格大于逻辑长度，否则 Expansion 不会
            // 记录 expansionDims，调度器会越界访问。
            int paddedLen = (std::max(totalLen, 1) / 128 + 1) * 128;
            std::vector<float> zeros((uint64_t)totalLen, 0.0f);
            for (int i = 0; i < std::min(blocks, (int)pastKeyValues.size()); i++) {
                Data key(DataType::FLOAT32, {1, totalLen, 1}, zeros);
                Data value(DataType::FLOAT32, {1, totalLen, 1}, zeros);
                key.SetKVCache();
                value.SetKVCache();
                key.Expansion({1, paddedLen, 1});
                value.Expansion({1, paddedLen, 1});
                pastKeyValues[i].first.FreeSpace();
                pastKeyValues[i].second.FreeSpace();
                pastKeyValues[i].first = Data();
                pastKeyValues[i].second = Data();
                pastKeyValues[i].first.CopyFrom(key);
                pastKeyValues[i].second.CopyFrom(value);
                pastKeyValues[i].first.SetKVCache();
                pastKeyValues[i].second.SetKVCache();
            }
        }

        // 调试：把张量以 float32 原始字节写到文件（FASTLLM_DSV41_DUMP_DIR）
        void V41DumpTensor(const Data &data, const std::string &name) {
            const char *dir = std::getenv("FASTLLM_DSV41_DUMP_DIR");
            if (dir == nullptr || dir[0] == '\0') {
                return;
            }
            Data cpu;
            cpu.CopyFrom(data);
            cpu.ToDevice(DataDevice::CPU);
            Data f32;
            if (cpu.dataType == DataType::FLOAT32) {
                f32.CopyFrom(cpu);
            } else if (cpu.dataType == DataType::INT32 || cpu.dataType == DataType::INT8) {
                f32 = Data(DataType::FLOAT32, cpu.dims);
                f32.Allocate();
                for (uint64_t i = 0; i < cpu.Count(0); i++) {
                    ((float*)f32.cpuData)[i] = cpu.dataType == DataType::INT32 ?
                        (float)((int32_t*)cpu.cpuData)[i] : (float)((uint8_t*)cpu.cpuData)[i];
                }
            } else {
                ToDataType(cpu, f32, DataType::FLOAT32);
            }
            f32.ToDevice(DataDevice::CPU);
            if (f32.cpuData == nullptr) {
                return;
            }
            std::string path = std::string(dir) + "/" + name + ".bin";
            FILE *fo = fopen(path.c_str(), "wb");
            if (fo != nullptr) {
                fwrite(f32.cpuData, 1, f32.GetBytes(), fo);
                fclose(fo);
            }
        }

        inline float V41Softplus(float x) {
            if (x > 20.0f) {
                return x;
            }
            if (x < -20.0f) {
                return std::exp(x);
            }
            return std::log1p(std::exp(x));
        }

        // ---------------- Engram 表 ----------------

        struct V41EngramTable {
            int64_t rows = 0;
            int dim = 0;
            int scaleBlock = 32;
            const uint8_t *data = nullptr;      // FP8 E4M3, [rows, dim]
            const uint8_t *scale = nullptr;     // UE8M0, [rows, dim / scaleBlock]
            std::vector<uint8_t> dataStorage;
            std::vector<uint8_t> scaleStorage;
            void *mmapData = nullptr;
            size_t mmapDataLen = 0;
            void *mmapScale = nullptr;
            size_t mmapScaleLen = 0;

            ~V41EngramTable() {
#if !defined(_WIN32) && !defined(_WIN64)
                if (mmapData != nullptr) {
                    munmap(mmapData, mmapDataLen);
                }
                if (mmapScale != nullptr) {
                    munmap(mmapScale, mmapScaleLen);
                }
#endif
            }
        };

        struct V41SafeTensorInfo {
            std::string fileName;
            std::string dtype;
            std::vector<int64_t> shape;
            uint64_t offset = 0;   // 绝对文件偏移
            uint64_t bytes = 0;
        };

        bool V41FindSafeTensor(const std::string &dir, const std::string &tensorName, V41SafeTensorInfo &info) {
            // 1. 通过 index 定位文件；没有 index 时尝试 model.safetensors
            std::vector<std::string> candidates;
            {
                std::ifstream fin(dir + "model.safetensors.index.json");
                if (fin.good()) {
                    std::stringstream ss;
                    ss << fin.rdbuf();
                    std::string err;
                    auto json = json11::Json::parse(ss.str(), err);
                    if (err.empty()) {
                        auto file = json["weight_map"][tensorName];
                        if (file.is_string()) {
                            candidates.push_back(dir + file.string_value());
                        }
                    }
                }
            }
            if (candidates.empty()) {
                candidates.push_back(dir + "model.safetensors");
            }
            for (auto &fileName : candidates) {
                std::ifstream fin(fileName, std::ios::binary);
                if (!fin.good()) {
                    continue;
                }
                uint64_t headerLen = 0;
                fin.read((char*)&headerLen, sizeof(headerLen));
                if (!fin.good() || headerLen == 0 || headerLen > (1ULL << 31)) {
                    continue;
                }
                std::string header(headerLen, '\0');
                fin.read(&header[0], headerLen);
                std::string err;
                auto json = json11::Json::parse(header, err);
                if (!err.empty()) {
                    continue;
                }
                auto item = json[tensorName];
                if (!item.is_object()) {
                    continue;
                }
                info.fileName = fileName;
                info.dtype = item["dtype"].string_value();
                info.shape.clear();
                for (auto &d : item["shape"].array_items()) {
                    info.shape.push_back((int64_t)d.number_value());
                }
                uint64_t st = (uint64_t)item["data_offsets"][0].number_value();
                uint64_t end = (uint64_t)item["data_offsets"][1].number_value();
                info.offset = 8 + headerLen + st;
                info.bytes = end - st;
                return true;
            }
            return false;
        }

        void V41ReadFileRange(const std::string &fileName, uint64_t offset, uint8_t *dst, uint64_t bytes) {
            FILE *fi = fopen(fileName.c_str(), "rb");
            AssertInFastLLM(fi != nullptr, "DeepSeekV41: can't open " + fileName);
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, offset, SEEK_SET);
#else
            fseeko(fi, (off_t)offset, SEEK_SET);
#endif
            const uint64_t chunk = 256ULL << 20;
            uint64_t done = 0;
            while (done < bytes) {
                uint64_t cur = std::min(chunk, bytes - done);
                uint64_t got = fread(dst + done, 1, cur, fi);
                AssertInFastLLM(got == cur, "DeepSeekV41: short read from " + fileName);
                done += cur;
            }
            fclose(fi);
        }

        bool V41MapFileRange(const std::string &fileName, uint64_t offset, uint64_t bytes,
                             void *&mapping, size_t &mapLen, const uint8_t *&ptr) {
#if defined(_WIN32) || defined(_WIN64)
            return false;
#else
            int fd = open(fileName.c_str(), O_RDONLY);
            if (fd < 0) {
                return false;
            }
            long pageSize = sysconf(_SC_PAGESIZE);
            uint64_t alignedOffset = offset / pageSize * pageSize;
            mapLen = (size_t)(bytes + (offset - alignedOffset));
            mapping = mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, (off_t)alignedOffset);
            close(fd);
            if (mapping == MAP_FAILED) {
                mapping = nullptr;
                return false;
            }
            ptr = (const uint8_t*)mapping + (offset - alignedOffset);
            return true;
#endif
        }

        bool V41IsPrime(int64_t x) {
            if (x < 2) {
                return false;
            }
            if (x % 2 == 0) {
                return x == 2;
            }
            for (int64_t d = 3; d * d <= x; d += 2) {
                if (x % d == 0) {
                    return false;
                }
            }
            return true;
        }

        inline float V41E8M0ToFloat(uint8_t v) {
            if (v == 0xFF) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return std::ldexp(1.0f, (int)v - 127);
        }
    }

    // ==================== 构造 / 参数 ====================

    DeepSeekV41Model::DeepSeekV41Model() {
        this->model_type = "deepseek_v41";
        this->model_struct = "deepseek_v41";
        this->canDoBatchForward = false;
        this->canDoConcurrentForward = true;
        this->defaultChunkedPrefillSize = 4096;

        weight.embeddingNames.clear();
        weight.embeddingNames.insert("embed.weight");
        weight.linearNames = {
            "head.weight",
            "layers.*.attn.wq_a.weight", "layers.*.attn.wq_b.weight",
            "layers.*.attn.wkv.weight",
            "layers.*.attn.wo_a.weight", "layers.*.attn.wo_b.weight",
            "layers.*.attn.indexer.wq_b.weight",
            "layers.*.attn.indexer.weights_proj.weight",
            "layers.*.attn.indexer.wk.weight",
            "layers.*.attn.compressor.wkv.weight",
            "layers.*.attn.compressor.wgate.weight",
            "layers.*.engram.wkv.weight",
            "layers.*.ffn.gate.weight",
            "layers.*.ffn.experts.*.w1.weight",
            "layers.*.ffn.experts.*.w2.weight",
            "layers.*.ffn.experts.*.w3.weight",
            "layers.*.ffn.shared_experts.w1.weight",
            "layers.*.ffn.shared_experts.w2.weight",
            "layers.*.ffn.shared_experts.w3.weight",
            "vision.patch_embed.proj.weight",
            "vision.blocks.*.attn.wqkv.weight", "vision.blocks.*.attn.wo.weight",
            "vision.blocks.*.mlp.w1.weight", "vision.blocks.*.mlp.w2.weight",
            "aligner.w1.weight", "aligner.w2.weight",
        };
    }

    DeepSeekV41Model::~DeepSeekV41Model() {
        ShutdownRuntime();
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41States.clear();
        }
        engramTables.clear();
    }

    void DeepSeekV41Model::InitParams() {
        // 1. 把 text_config.* 展平到顶层（HF config 为嵌套结构）
        {
            std::vector<std::pair<std::string, std::string> > flattened;
            for (auto &it : this->weight.dicts) {
                if (V41StartsWith(it.first, "text_config.")) {
                    flattened.push_back({it.first.substr(strlen("text_config.")), it.second});
                }
            }
            for (auto &kv : flattened) {
                if (!V41HasKey(this->weight, kv.first)) {
                    this->weight.AddDict(kv.first, kv.second);
                }
            }
            if (!V41HasKey(this->weight, "rope_scaling.type") && V41HasKey(this->weight, "rope_scaling.rope_type")) {
                this->weight.AddDict("rope_scaling.type", this->weight.dicts["rope_scaling.rope_type"]);
            }
            // V4.1 没有 hash 路由层
            if (!V41HasKey(this->weight, "num_hash_layers")) {
                this->weight.AddDict("num_hash_layers", "0");
            }
        }

        // 2. 复用 V4 的基础解析（尺寸、RoPE、MoE 合并规则、特殊权重注册等）
        DeepSeekV4Model::InitParams();
        this->rms_norm_eps = V41DictFloat(this->weight, "rms_norm_eps", this->rms_norm_eps);

        // 3. V4.1 专有参数
        kv_source_layer_ids = V41DictIntArray(this->weight, "kv_source_layer_ids");
        index_source_layer_ids = V41DictIntArray(this->weight, "index_source_layer_ids");
        candidate_source_layer_id = V41DictInt(this->weight, "candidate_source_layer_id", -1);
        candidate_topk_blocks = V41DictInt(this->weight, "candidate_topk_blocks", 0);
        candidate_block_size = V41DictInt(this->weight, "candidate_block_size", 0);
        gate_temp = V41DictFloat(this->weight, "gate_temp", 1.0f);
        image_token_id = V41DictInt(this->weight, "image_token_id", -1);

        engram_layer_ids = V41DictIntArray(this->weight, "engram_layer_ids");
        engram_num_embeddings = V41DictInt64Array(this->weight, "engram_num_embeddings");
        engram_max_ngram_size = V41DictInt(this->weight, "engram_max_ngram_size", 4);
        engram_vocab_size = V41DictInt(this->weight, "engram_vocab_size", 0);
        engram_n_heads = V41DictInt(this->weight, "engram_n_heads", 0);
        engram_head_dim = V41DictInt(this->weight, "engram_head_dim", 0);
        engram_pad_token_id = V41DictInt(this->weight, "engram_pad_token_id", 2);
        engram_compressed_vocab_size = V41DictInt(this->weight, "engram_compressed_vocab_size", 0);

        // 4. 每层的跨层共享关系
        kvSourceOf.assign(block_cnt, -1);
        indexSourceOf.assign(block_cnt, -1);
        isKvSource.assign(block_cnt, 0);
        isIndexSource.assign(block_cnt, 0);
        int curKv = -1, curIndex = -1;
        for (int layer = 0; layer < block_cnt; layer++) {
            int ratio = compress_ratios.size() > (size_t)layer ? compress_ratios[layer] : 0;
            if (V41Contains(kv_source_layer_ids, layer)) {
                curKv = layer;
                isKvSource[layer] = 1;
            }
            if (V41Contains(index_source_layer_ids, layer)) {
                curIndex = layer;
                isIndexSource[layer] = 1;
            }
            if (ratio > 0) {
                AssertInFastLLM(curKv >= 0 && curIndex >= 0 &&
                                compress_ratios[curKv] == ratio,
                                "DeepSeekV41: layer " + std::to_string(layer) +
                                " uses compressed attention but has no matching kv/index source layer.");
                kvSourceOf[layer] = curKv;
                indexSourceOf[layer] = curIndex;
            }
            AssertInFastLLM(ratio == 0 || ratio == 1 || ratio == 2,
                            "DeepSeekV41: unsupported compress ratio " + std::to_string(ratio));
        }
        for (int layer : kv_source_layer_ids) {
            AssertInFastLLM(layer >= 0 && layer < block_cnt && V41Contains(index_source_layer_ids, layer),
                            "DeepSeekV41: every kv source layer must also be an index source layer.");
        }

        // 5. 这些小权重保持源精度
        for (int i = 0; i < block_cnt; i++) {
            std::string pre = "layers." + std::to_string(i);
            this->cantQuantLinears.insert(pre + ".attn.compressor.wkv.weight");
            this->cantQuantLinears.insert(pre + ".attn.compressor.wgate.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.wk.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.weights_proj.weight");
            this->cantQuantLinears.insert(pre + ".ffn.gate.weight");
        }

        LoadEngramMeta();
        InitVisionParams();

        printf("[Fastllm] DeepSeek-V4.1: %d layers, %d experts (top-%d), kv sources = %d, index sources = %d, "
               "engram layers = %d%s, vision layers = %d\n",
               block_cnt, num_experts, num_experts_per_tok, (int)kv_source_layer_ids.size(),
               (int)index_source_layer_ids.size(), (int)engram_layer_ids.size(),
               engramMeta.loaded ? "" : " (engram meta NOT loaded)", vision_n_layers);
        fflush(stdout);
    }

    // ==================== Engram 元数据 ====================

    void DeepSeekV41Model::BuildEngramPrimes() {
        // 与 engram.py::EngramLayout.from_args 一致：所有层共享一个 seen 集合，
        // 逐层、逐 n-gram 大小、逐 head 取"下一个未使用的素数"。
        engramMeta.primes.clear();
        engramMeta.offsets.clear();
        std::set<int64_t> seen;
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            std::vector<int64_t> flat;
            for (int n = 0; n < engram_max_ngram_size - 1; n++) {
                int64_t current = (int64_t)engram_vocab_size - 1;
                for (int h = 0; h < engram_n_heads; h++) {
                    int64_t candidate = current + 1;
                    while (!V41IsPrime(candidate) || seen.count(candidate)) {
                        candidate++;
                    }
                    seen.insert(candidate);
                    current = candidate;
                    flat.push_back(candidate);
                }
            }
            std::vector<int64_t> offsets(flat.size(), 0);
            int64_t total = 0;
            for (size_t i = 0; i < flat.size(); i++) {
                offsets[i] = total;
                total += flat[i];
            }
            if (l < engram_num_embeddings.size()) {
                AssertInFastLLM(total == engram_num_embeddings[l],
                                "DeepSeekV41: engram prime layout mismatch (layer " + std::to_string(l) +
                                ": " + std::to_string(total) + " vs " + std::to_string(engram_num_embeddings[l]) + ").");
            }
            engramMeta.primes.push_back(flat);
            engramMeta.offsets.push_back(offsets);
        }
    }

    void DeepSeekV41Model::LoadEngramMeta() {
        engramMeta.loaded = false;
        if (engram_layer_ids.empty()) {
            return;
        }
        std::string metaPath;
        if (const char *env = std::getenv("FASTLLM_DSV41_ENGRAM_META")) {
            metaPath = env;
        } else if (V41HasKey(this->weight, "engram_meta_path")) {
            metaPath = this->weight.dicts["engram_meta_path"];
        } else if (V41HasKey(this->weight, "model_directory")) {
            metaPath = this->weight.dicts["model_directory"] + "engram_meta.json";
        }
        if (metaPath.empty()) {
            return;
        }
        std::ifstream fin(metaPath);
        if (!fin.good()) {
            printf("[Fastllm] DeepSeek-V4.1: engram meta file not found: %s\n", metaPath.c_str());
            return;
        }
        std::stringstream ss;
        ss << fin.rdbuf();
        std::string err;
        auto json = json11::Json::parse(ss.str(), err);
        AssertInFastLLM(err.empty() && json.is_object(), "DeepSeekV41: failed to parse engram meta " + metaPath);
        engramMeta.tokenMap.clear();
        for (auto &v : json["token_map"].array_items()) {
            engramMeta.tokenMap.push_back(v.int_value());
        }
        engramMeta.compressedVocabSize = json["compressed_vocab_size"].int_value();
        AssertInFastLLM(engram_compressed_vocab_size == 0 ||
                        engramMeta.compressedVocabSize == engram_compressed_vocab_size,
                        "DeepSeekV41: engram compressed vocab size mismatch (" +
                        std::to_string(engramMeta.compressedVocabSize) + " vs " +
                        std::to_string(engram_compressed_vocab_size) + ").");
        engramMeta.multipliers.clear();
        for (auto &row : json["multipliers"].array_items()) {
            std::vector<int64_t> values;
            for (auto &v : row.array_items()) {
                // JSON 数字用 double 存放，multiplier 上界约 2^63 / vocab / 2，精度不够；
                // 因此 Python 侧以字符串写出，这里兼容两种写法
                if (v.is_string()) {
                    values.push_back(std::strtoll(v.string_value().c_str(), nullptr, 10));
                } else {
                    values.push_back((int64_t)v.number_value());
                }
            }
            engramMeta.multipliers.push_back(values);
        }
        AssertInFastLLM(engramMeta.multipliers.size() == engram_layer_ids.size() &&
                        (int)engramMeta.tokenMap.size() > engram_pad_token_id,
                        "DeepSeekV41: engram meta is incomplete.");
        engramMeta.padCompressedId = engramMeta.tokenMap[engram_pad_token_id];
        BuildEngramPrimes();
        engramMeta.loaded = true;
    }

    // ==================== 权重映射 ====================

    std::map<std::string, std::vector<std::pair<std::string, DataType> > >
    DeepSeekV41Model::GetTensorMap(const std::vector<std::string> &tensorNames) {
        std::map<std::string, std::vector<std::pair<std::string, DataType> > > result;
        std::vector<std::string> ordinary;
        for (const std::string &name : tensorNames) {
            // DSpark 草稿层暂不加载
            if (V41StartsWith(name, "mtp.")) {
                continue;
            }
            // 视觉编码器：线性层权重走通用映射（float16），其余（norm / bias / 分隔符嵌入）保持 float32
            if (IsVisionTensor(name)) {
                if (!VisionEnabled()) {
                    continue;
                }
                if (V41EndsWith(name, ".weight") && this->weight.GetWeightType(name) == WeightType::LINEAR) {
                    ordinary.push_back(name);
                } else {
                    result[name].push_back({name, DataType::FLOAT32});
                }
                continue;
            }
            // Engram 表由模型自行读取（超出通用加载器的 int32 scale 索引范围）
            if (name.find(".engram.embed.") != std::string::npos) {
                continue;
            }
            if (name.find(".engram.q_weight") != std::string::npos ||
                name.find(".engram.k_weight") != std::string::npos ||
                V41EndsWith(name, ".attn_sink") ||
                name.find(".ffn.gate.bias") != std::string::npos) {
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            if (V41EndsWith(name, ".ffn.gate.weight") ||
                name.find(".attn.compressor.wkv.weight") != std::string::npos ||
                name.find(".attn.compressor.wgate.weight") != std::string::npos) {
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            if (name.find(".attn.indexer.wk.weight") != std::string::npos ||
                name.find(".attn.indexer.weights_proj.weight") != std::string::npos) {
                result[name].push_back({name, DataType::BFLOAT16});
                continue;
            }
            ordinary.push_back(name);
        }
        auto mapped = basellm::GetTensorMap(ordinary);
        for (auto &it : mapped) {
            result[it.first] = it.second;
        }
        return result;
    }

    void DeepSeekV41Model::OnModelWeightsLoaded() {
        if (engram_layer_ids.empty()) {
            return;
        }
        AssertInFastLLM(V41HasKey(this->weight, "model_directory"),
                        "DeepSeekV41: model directory is unknown, can't load engram tables.");
        std::string dir = this->weight.dicts["model_directory"];
        bool useMmap = V41EnvFlag("FASTLLM_DSV41_ENGRAM_MMAP");
        engramTables.clear();
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            int layer = engram_layer_ids[l];
            std::string base = "layers." + std::to_string(layer) + ".engram.embed.";
            V41SafeTensorInfo weightInfo, scaleInfo;
            AssertInFastLLM(V41FindSafeTensor(dir, base + "weight", weightInfo) &&
                            V41FindSafeTensor(dir, base + "scale", scaleInfo),
                            "DeepSeekV41: can't locate engram table " + base + "weight in " + dir);
            AssertInFastLLM(weightInfo.dtype == "F8_E4M3" && weightInfo.shape.size() == 2 &&
                            (scaleInfo.dtype == "F8_E8M0" || scaleInfo.dtype == "U8") &&
                            scaleInfo.shape.size() == 2 && scaleInfo.shape[0] == weightInfo.shape[0] &&
                            weightInfo.shape[1] % scaleInfo.shape[1] == 0,
                            "DeepSeekV41: unsupported engram table format for " + base + "weight");
            auto table = std::make_shared<V41EngramTable>();
            table->rows = weightInfo.shape[0];
            table->dim = (int)weightInfo.shape[1];
            table->scaleBlock = (int)(weightInfo.shape[1] / scaleInfo.shape[1]);
            AssertInFastLLM(table->dim == engram_head_dim && weightInfo.bytes == (uint64_t)table->rows * table->dim &&
                            scaleInfo.bytes == (uint64_t)table->rows * (table->dim / table->scaleBlock),
                            "DeepSeekV41: engram table byte count mismatch for " + base + "weight");
            printf("[Fastllm] DeepSeek-V4.1: loading engram table for layer %d (%.1f GB, %s)...\n",
                   layer, (weightInfo.bytes + scaleInfo.bytes) / 1e9, useMmap ? "mmap" : "resident");
            fflush(stdout);
            bool mapped = false;
            if (useMmap) {
                mapped = V41MapFileRange(weightInfo.fileName, weightInfo.offset, weightInfo.bytes,
                                         table->mmapData, table->mmapDataLen, table->data) &&
                         V41MapFileRange(scaleInfo.fileName, scaleInfo.offset, scaleInfo.bytes,
                                         table->mmapScale, table->mmapScaleLen, table->scale);
            }
            if (!mapped) {
                table->dataStorage.resize(weightInfo.bytes);
                table->scaleStorage.resize(scaleInfo.bytes);
                V41ReadFileRange(weightInfo.fileName, weightInfo.offset, table->dataStorage.data(), weightInfo.bytes);
                V41ReadFileRange(scaleInfo.fileName, scaleInfo.offset, table->scaleStorage.data(), scaleInfo.bytes);
                table->data = table->dataStorage.data();
                table->scale = table->scaleStorage.data();
            }
            engramTables.push_back(std::static_pointer_cast<void>(table));
        }
        printf("[Fastllm] DeepSeek-V4.1: engram tables ready.\n");
        fflush(stdout);
    }

    // ==================== Engram 前向 ====================

    void DeepSeekV41Model::ComputeEngramHashes(int engramLayerIndex,
                                               const std::vector<int> &history, int startPos, int seqlen,
                                               std::vector<int64_t> &rows) const {
        const int maxNgram = engram_max_ngram_size;
        const int heads = engram_n_heads;
        const int cols = (maxNgram - 1) * heads;
        const auto &multipliers = engramMeta.multipliers[engramLayerIndex];
        const auto &primes = engramMeta.primes[engramLayerIndex];
        const auto &offsets = engramMeta.offsets[engramLayerIndex];
        rows.assign((size_t)seqlen * cols, 0);
        std::vector<int64_t> tokens(maxNgram);
        for (int i = 0; i < seqlen; i++) {
            int pos = startPos + i;
            bool blocked = false;
            for (int shift = 0; shift < maxNgram; shift++) {
                int p = pos - shift;
                int source = p >= 0 ? history[p] : -1;
                blocked = blocked || p < 0 || source < 0;
                tokens[shift] = blocked ? engramMeta.padCompressedId : source;
            }
            // rolling XOR：第 i 步之后的值是 (i+1)-gram 的哈希
            uint64_t rolling = (uint64_t)tokens[0] * (uint64_t)multipliers[0];
            for (int n = 1; n < maxNgram; n++) {
                rolling ^= (uint64_t)tokens[n] * (uint64_t)multipliers[n];
                int64_t signedRolling = (int64_t)rolling;
                for (int h = 0; h < heads; h++) {
                    int col = (n - 1) * heads + h;
                    int64_t prime = primes[col];
                    int64_t bucket = signedRolling % prime;
                    if (bucket < 0) {
                        bucket += prime;   // Python 取模语义
                    }
                    rows[(size_t)i * cols + col] = offsets[col] + bucket;
                }
            }
        }
    }

    void DeepSeekV41Model::GatherEngramRows(int layer, const std::vector<int64_t> &rows, int tokens, Data &output) {
        int engramLayerIndex = -1;
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            if (engram_layer_ids[l] == layer) {
                engramLayerIndex = (int)l;
            }
        }
        AssertInFastLLM(engramLayerIndex >= 0 && engramLayerIndex < (int)engramTables.size(),
                        "DeepSeekV41: engram table for layer " + std::to_string(layer) + " is not loaded.");
        const V41EngramTable &table = *std::static_pointer_cast<V41EngramTable>(engramTables[engramLayerIndex]);
        const int cols = (int)(rows.size() / std::max(1, tokens));
        const int dim = table.dim;
        const int scaleCols = dim / table.scaleBlock;
        output.ToDevice(DataDevice::CPU);
        output = Data(DataType::BFLOAT16, {1, tokens, cols * dim});
        output.Allocate(false);
        uint16_t *dst = (uint16_t*)output.cpuData;
        static const FP8E4M3ToFP32Manager fp8;

        auto worker = [&](int st, int end) {
            for (int t = st; t < end; t++) {
                for (int c = 0; c < cols; c++) {
                    int64_t row = rows[(size_t)t * cols + c];
                    AssertInFastLLM(row >= 0 && row < table.rows, "DeepSeekV41: engram hash out of range.");
                    const uint8_t *src = table.data + (uint64_t)row * dim;
                    const uint8_t *sc = table.scale + (uint64_t)row * scaleCols;
                    uint16_t *out = dst + ((uint64_t)t * cols + c) * dim;
                    for (int d = 0; d < dim; d++) {
                        float v = fp8.dict[src[d]] * V41E8M0ToFloat(sc[d / table.scaleBlock]);
                        out[d] = Float32ToBFloat16RNEBits(v);
                    }
                }
            }
        };
        int threads = std::min(tokens, std::max(1, (int)std::thread::hardware_concurrency() / 2));
        threads = std::min(threads, 32);
        if (threads <= 1 || tokens < 8) {
            worker(0, tokens);
        } else {
            std::vector<std::thread> pool;
            int per = (tokens + threads - 1) / threads;
            for (int i = 0; i < threads; i++) {
                int st = i * per, end = std::min(tokens, st + per);
                if (st < end) {
                    pool.emplace_back(worker, st, end);
                }
            }
            for (auto &th : pool) {
                th.join();
            }
        }
    }

    void DeepSeekV41Model::RunEngram(int layer, int engramLayerIndex, const std::vector<int> &history,
                                     int startPos, int seqlen, Data &hiddenStates) {
        AssertInFastLLM(engramMeta.loaded,
                        "DeepSeekV41: engram meta is not loaded. Generate engram_meta.json with "
                        "`python -m ftllm.deepseek_v41_engram <model_dir>` or set FASTLLM_DSV41_ENGRAM_META.");
        std::string pre = "layers." + std::to_string(layer) + ".engram";
        std::vector<int64_t> rows;
        ComputeEngramHashes(engramLayerIndex, history, startPos, seqlen, rows);
        Data gathered;
        GatherEngramRows(layer, rows, seqlen, gathered);
        Data kv;
        Linear(gathered, weight[pre + ".wkv.weight"], Data(), kv);
        Data mask;
        bool hasDead = false;
        std::vector<float> maskValues(seqlen, 1.0f);
        for (int i = 0; i < seqlen; i++) {
            if (history[startPos + i] < 0) {
                maskValues[i] = 0.0f;
                hasDead = true;
            }
        }
        if (hasDead) {
            mask.CopyFrom(Data(DataType::FLOAT32, {1, seqlen}, maskValues));
        }
        V41EngramApply(hiddenStates, kv, weight[pre + ".q_weight"], weight[pre + ".k_weight"],
                       hasDead ? &mask : nullptr, rms_norm_eps);
    }

    // ==================== 请求状态 ====================

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::GetOrCreateState(
            std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset) {
        const void *key = (const void*)&pastKeyValues;
        std::lock_guard<std::mutex> guard(v41StateMutex);
        auto it = v41States.find(key);
        if (it != v41States.end() && !reset) {
            return it->second;
        }
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(block_cnt);
        v41States[key] = state;
        return state;
    }

    void DeepSeekV41Model::OnResponseContextCreated(ResponseContext *context) {
        // 图文请求：调度器可能只用普通 Forward 逐块 prefill，因此在这里就把多模态输入记到请求状态里，
        // 由第一个 prefill 块编码图像（见 ForwardSingle）
        if (context != nullptr && !context->multimodalInput.empty()) {
            GetOrCreateState(context->pastKeyValues, false)->pendingMultimodal = &context->multimodalInput;
        }
    }

    void DeepSeekV41Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(v41StateMutex);
        v41States.erase((const void*)&context->pastKeyValues);
    }

    void DeepSeekV41Model::TryRecordResponseContext(ResponseContext *context) {
        (void)context;
    }

    bool DeepSeekV41Model::TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) {
        (void)inputTokens;
        cacheLen = 0;
        return false;
    }

    void DeepSeekV41Model::TryRecordHistoryCache(const std::vector<int> &allTokens) {
        (void)allTokens;
    }

    // ==================== 前向 ====================

    int DeepSeekV41Model::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                                  std::vector<std::pair<Data, Data> > &pastKeyValues,
                                  const GenerationConfig &generationConfig,
                                  const LastTokensManager &lastTokens,
                                  std::vector<float> *retLogits) {
        std::vector<std::vector<float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues,
                            generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector<int> DeepSeekV41Model::ForwardBatch(int batch, const Data &inputIds,
                                                    const std::vector<Data*> &attentionMask,
                                                    const std::vector<Data*> &positionIds,
                                                    const std::vector<int> &seqLens,
                                                    std::vector<std::pair<Data*, Data*> > &pastKeyValues,
                                                    const std::vector<GenerationConfig> &generationConfigs,
                                                    const LastTokensManager &lastTokens,
                                                    std::vector<std::vector<float>*> *retLogits) {
        (void)attentionMask; (void)positionIds; (void)seqLens; (void)pastKeyValues;
        (void)generationConfigs; (void)lastTokens; (void)retLogits; (void)batch; (void)inputIds;
        ErrorInFastLLM("DeepSeekV41Model: multi-request batched forward is not supported yet "
                       "(canDoBatchForward is false, the scheduler should call Forward per request).");
        return {};
    }

    std::vector<int> DeepSeekV41Model::ForwardBatch(int batch, const Data &inputIds, const Data &attentionMask,
                                                    const Data &positionIds,
                                                    std::vector<std::pair<Data, Data> > &pastKeyValues,
                                                    const GenerationConfig &generationConfig,
                                                    const LastTokensManager &lastTokens,
                                                    std::vector<std::vector<float>*> *retLogits) {
        (void)attentionMask;
        AssertInFastLLM(batch == 1 && inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "DeepSeekV41Model::ForwardBatch only supports one sequence per call.");
        return ForwardSingle(inputIds, positionIds, pastKeyValues, generationConfig, lastTokens, retLogits,
                             nullptr, nullptr);
    }

    std::vector<int> DeepSeekV41Model::ForwardSingle(const Data &inputIds, const Data &positionIds,
                                                     std::vector<std::pair<Data, Data> > &pastKeyValues,
                                                     const GenerationConfig &generationConfig,
                                                     const LastTokensManager &lastTokens,
                                                     std::vector<std::vector<float>*> *retLogits,
                                                     const Data *inputEmbeds,
                                                     const std::vector<int> *imageMask) {
        const int seqlen = inputIds.dims[1];
        AssertInFastLLM(inputEmbeds == nullptr ||
                        (inputEmbeds->dims.size() == 3 && inputEmbeds->dims[1] == seqlen &&
                         inputEmbeds->dims[2] == embed_dim),
                        "DeepSeekV41Model: inputEmbeds must be [1, seqlen, dim].");
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model: imageMask length mismatch.");
        auto hasAnyImageToken = [](const std::vector<int> *mask) {
            if (mask == nullptr) {
                return false;
            }
            for (int v : *mask) {
                if (v != 0) {
                    return true;
                }
            }
            return false;
        };
        int startPos = 0;
        if (positionIds.dims.size() >= 1 && positionIds.Count(0) > 0) {
            auto pids = V41ReadTokenIds(positionIds);
            startPos = pids.empty() ? 0 : pids[0];
        }
        // 新请求从位置 0 开始：已用过的状态要重建，尚未使用的状态（可能带有图文请求的多模态输入）保留
        auto state = GetOrCreateState(pastKeyValues, false);
        if (startPos == 0 && state->totalLen > 0) {
            state = GetOrCreateState(pastKeyValues, true);
        }
        AssertInFastLLM(state->totalLen == startPos,
                        "DeepSeekV41Model: position mismatch (cache has " + std::to_string(state->totalLen) +
                        " tokens, request starts at " + std::to_string(startPos) + ").");
        std::vector<int> tokenIds = V41ReadTokenIds(inputIds);
        // 图文请求：第一个块编码全部图像，之后每个块把与图像 span 重叠的位置换成图像嵌入
        Data imageEmbeds;
        std::vector<int> imageMaskStorage;
        if (inputEmbeds == nullptr && state->pendingMultimodal != nullptr) {
            if (!state->imagesEncoded) {
                EncodeImageSpans(*state->pendingMultimodal, *state);
            }
            if (PrepareImageEmbeds(inputIds, startPos, *state, imageEmbeds, imageMaskStorage)) {
                inputEmbeds = &imageEmbeds;
                imageMask = &imageMaskStorage;
            }
        }
        const bool hasImageTokens = hasAnyImageToken(imageMask);
        if (!engram_layer_ids.empty()) {
            for (int tok : tokenIds) {
                int compressed = -1;
                if (tok != image_token_id && engramMeta.loaded && tok >= 0 &&
                    tok < (int)engramMeta.tokenMap.size()) {
                    compressed = engramMeta.tokenMap[tok];
                }
                state->engramHistory.push_back(compressed);
            }
        }

        const int dim = embed_dim;
        const int headDim = head_dim_full;
        const float softmaxScale = 1.0f / std::sqrt((float)headDim);
        const int indexTopK = index_topk;
        V41RopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                    rope_scaling_beta_fast, rope_scaling_beta_slow};
        V41RopeParams compressRope = {qk_rope_head_dim, compress_rope_theta,
                                      (int)rope_scaling_original_max_position_embeddings, rope_factor,
                                      rope_scaling_beta_fast, rope_scaling_beta_slow};

        // MoE 权重表（首次构建）
        if (weights.empty()) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            auto getWeightPtr = [&](const std::string &name) -> Data* {
                auto it = weight.weight.find(name);
                return it == weight.weight.end() ? nullptr : &it->second;
            };
            for (int layer = 0; layer < block_cnt; layer++) {
                std::string pre = "layers." + std::to_string(layer) + ".ffn";
                weights[layer].push_back(getWeightPtr(pre + ".shared_experts.gateup.weight"));
                weights[layer].push_back(getWeightPtr(pre + ".shared_experts.w2.weight"));
                biass[layer].push_back(nullptr);
                biass[layer].push_back(nullptr);
                for (int expert = 0; expert < num_experts; expert++) {
                    weights[layer].push_back(getWeightPtr(pre + ".experts." + std::to_string(expert) + ".gateup.weight"));
                    weights[layer].push_back(getWeightPtr(pre + ".experts." + std::to_string(expert) + ".w2.weight"));
                    biass[layer].push_back(nullptr);
                    biass[layer].push_back(nullptr);
                }
            }
        }

        // ---- embedding -> hc 份 ----
        Data hiddenStates, hiddenTemp;
        {
            Data embedOut;
            if (inputEmbeds != nullptr) {
                embedOut.CopyFrom(*inputEmbeds);
            } else {
                Embedding(inputIds, weight["embed.weight"], embedOut);
            }
            ToDataType(embedOut, DataType::BFLOAT16);
            embedOut.Reshape({1, seqlen, 1, dim});
            Repeat(embedOut, 2, hc_mult, hiddenStates);
        }
        Data *curHidden = &hiddenStates;
        Data *nextHidden = &hiddenTemp;
        const bool dumpDebug = std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr;
        const std::string dumpSuffix = startPos == 0 ? std::string("") : "_p" + std::to_string(startPos);
        if (dumpDebug) {
            V41DumpTensor(hiddenStates, "fl_embed" + dumpSuffix);
        }

        Data preMix;
        {
            std::vector<float> values((uint64_t)seqlen * hc_mult, 0.0f);
            for (int i = 0; i < seqlen; i++) {
                values[(uint64_t)i * hc_mult] = 1.0f;
            }
            preMix.CopyFrom(Data(DataType::FLOAT32, {1, seqlen, hc_mult}, values));
        }

        // 本次前向内跨层共享的 indexer 结果
        Data sharedTopK;
        Data candidateMask;
        bool hasCandidates = false;

        Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
        Data x, attnInput, qr, qNorm, q, kv, attnOut, woAOut, attnProj;
        Data ffnInput, ffnOut, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;

        for (int layer = 0; layer < block_cnt; layer++) {
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            std::string pre = "layers." + std::to_string(layer);
            const int ratio = compress_ratios[layer];
            const V41RopeParams &rope = ratio > 0 ? compressRope : windowRope;
            DeepSeekV41LayerCache &cache = state->layers[layer];

            // ---- Engram ----
            for (size_t l = 0; l < engram_layer_ids.size(); l++) {
                if (engram_layer_ids[l] == layer) {
                    RunEngram(layer, (int)l, state->engramHistory, startPos, seqlen, *curHidden);
                }
            }

            // ---- attention ----
            V41HcMix(*curHidden, weight[pre + ".hc_attn_fn"], weight[pre + ".hc_attn_scale"],
                     weight[pre + ".hc_attn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                     attnPre, attnPost, attnComb);
            V41HcApplyPre(*curHidden, preMix, x);
            V41RMSNormBF16(x, weight[pre + ".attn_norm.weight"], rms_norm_eps, attnInput);

            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr);
            V41RMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({1, seqlen, num_attention_heads, headDim});
            V41RotaryQuant(q, rope, startPos, 1, false, 0, 32);

            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv);
            V41RMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            kv.Reshape({1, seqlen, headDim});
            V41RotaryQuant(kv, rope, startPos, 1, false, 1, 32);

            Data *compressedKV = nullptr;
            Data *cmpIdx = nullptr;
            if (ratio > 0) {
                const int src = kvSourceOf[layer];
                DeepSeekV41LayerCache &srcCache = state->layers[src];
                if (isKvSource[layer]) {
                    std::string cpre = pre + ".attn.compressor";
                    Data xFloat, rawKV, rawScore, allKV, allScore;
                    ToDataType(attnInput, xFloat, DataType::FLOAT32);
                    Linear(xFloat, weight[cpre + ".wkv.weight"], Data(), rawKV);
                    ToDataType(rawKV, DataType::FLOAT32);
                    if (ratio > 1) {
                        Linear(xFloat, weight[cpre + ".wgate.weight"], Data(), rawScore);
                        ToDataType(rawScore, DataType::FLOAT32);
                    }
                    Data *kvAll = &rawKV, *scoreAll = &rawScore;
                    if (cache.rawTail > 0) {
                        Cat(cache.rawTailKV, rawKV, 1, allKV);
                        kvAll = &allKV;
                        if (ratio > 1) {
                            Cat(cache.rawTailScore, rawScore, 1, allScore);
                            scoreAll = &allScore;
                        }
                    }
                    const int n = kvAll->dims[1];
                    const int full = n - n % ratio;
                    const int blocks = full / ratio;
                    if (blocks > 0) {
                        Data kvPart, scorePart, latent;
                        Data *kvPartPtr = kvAll, *scorePartPtr = ratio > 1 ? scoreAll : nullptr;
                        if (full != n) {
                            Split(*kvAll, 1, 0, full, kvPart);
                            kvPartPtr = &kvPart;
                            if (ratio > 1) {
                                Split(*scoreAll, 1, 0, full, scorePart);
                                scorePartPtr = &scorePart;
                            }
                        }
                        V41Compress(*kvPartPtr, scorePartPtr, weight[cpre + ".norm.weight"], ratio, rms_norm_eps, latent);
                        const int blockStart = cache.compressedBlocks;
                        // indexer key 由 pre-RoPE latent 派生
                        if (isIndexSource[layer]) {
                            std::string ipre = pre + ".attn.indexer";
                            Data kIdx;
                            Linear(latent, weight[ipre + ".wk.weight"], Data(), kIdx);
                            V41RMSNormBF16(kIdx, weight[ipre + ".k_norm.weight"], rms_norm_eps, kIdx);
                            kIdx.Reshape({1, blocks, index_head_dim});
                            V41RotaryQuant(kIdx, rope, blockStart * ratio, ratio, false, 2, 32);
                            V41AppendRows(cache.indexK, kIdx);
                        }
                        V41RotaryQuant(latent, rope, blockStart * ratio, ratio, false, 3, 16);
                        V41AppendRows(cache.compressedKV, latent);
                        cache.compressedBlocks += blocks;
                    }
                    const int rem = n - full;
                    if (rem > 0) {
                        Data tailKV, tailScore;
                        Split(*kvAll, 1, full, n, tailKV);
                        cache.rawTailKV.CopyFrom(tailKV);
                        if (ratio > 1) {
                            Split(*scoreAll, 1, full, n, tailScore);
                            cache.rawTailScore.CopyFrom(tailScore);
                        }
                    }
                    cache.rawTail = rem;
                }
                AssertInFastLLM(srcCache.compressedBlocks == (startPos + seqlen) / ratio,
                                "DeepSeekV41: compressed cache is out of sync at layer " + std::to_string(layer));
                if (srcCache.compressedBlocks > 0) {
                    compressedKV = &srcCache.compressedKV;
                    if (isIndexSource[layer]) {
                        std::string ipre = pre + ".attn.indexer";
                        Data qIdx, idxWeights, idxWeightsScaled, score;
                        Linear(qNorm, weight[ipre + ".wq_b.weight"], Data(), qIdx);
                        qIdx.Reshape({1, seqlen, index_n_heads, index_head_dim});
                        V41RotaryQuant(qIdx, rope, startPos, 1, false, 2, 32);
                        Linear(attnInput, weight[ipre + ".weights_proj.weight"], Data(), idxWeights);
                        ToDataType(idxWeights, DataType::FLOAT32);
                        Mul(idxWeights, (1.0f / std::sqrt((float)index_head_dim)) * (1.0f / std::sqrt((float)index_n_heads)),
                            idxWeightsScaled);
                        V41IndexerScore(qIdx, idxWeightsScaled, srcCache.indexK, score);
                        if (dumpDebug) {
                            V41DumpTensor(qIdx, "fl_layer" + std::to_string(layer) + "_idxq" + dumpSuffix);
                            V41DumpTensor(idxWeightsScaled, "fl_layer" + std::to_string(layer) + "_idxw" + dumpSuffix);
                            V41DumpTensor(score, "fl_layer" + std::to_string(layer) + "_score" + dumpSuffix);
                        }
                        if (layer == candidate_source_layer_id && candidate_block_size > 0 && candidate_topk_blocks > 0) {
                            V41CandidateBlocks(score, candidate_block_size, candidate_topk_blocks, ratio, startPos,
                                               candidateMask);
                            hasCandidates = true;
                            if (dumpDebug) {
                                V41DumpTensor(candidateMask, "fl_layer" + std::to_string(layer) + "_cand" + dumpSuffix);
                            }
                        }
                        const bool useCandidates = hasCandidates && candidate_source_layer_id >= 0 &&
                                                   candidate_source_layer_id < layer;
                        V41IndexerTopK(score, useCandidates ? &candidateMask : nullptr, indexTopK, ratio, startPos,
                                       std::max(1, candidate_block_size), sharedTopK);
                    }
                    if (sharedTopK.dims.size() == 3) {
                        cmpIdx = &sharedTopK;
                    }
                }
            }

            V41SparseAttention(q, kv, startPos > 0 ? &cache.windowKV : nullptr, compressedKV, cmpIdx,
                               weight[pre + ".attn.attn_sink"], window_size, startPos, softmaxScale, attnOut);
            if (dumpDebug) {
                std::string tag = "fl_layer" + std::to_string(layer);
                V41DumpTensor(q, tag + "_q" + dumpSuffix);
                V41DumpTensor(kv, tag + "_kv" + dumpSuffix);
                V41DumpTensor(attnOut, tag + "_attn_o_raw" + dumpSuffix);
                if (compressedKV != nullptr) {
                    V41DumpTensor(*compressedKV, tag + "_ckv" + dumpSuffix);
                }
                if (ratio > 0 && isKvSource[layer]) {
                    V41DumpTensor(cache.indexK, tag + "_idxk" + dumpSuffix);
                }
                if (startPos > 0) {
                    V41DumpTensor(cache.windowKV, tag + "_ring" + dumpSuffix);
                }
            }
            V41WindowStore(kv, cache.windowKV, startPos, window_size);
            V41RotaryQuant(attnOut, rope, startPos, 1, true, 0, 32);
            DeepSeekV4WoA(attnOut, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnProj);
            if (dumpDebug) {
                V41DumpTensor(attnInput, "fl_layer" + std::to_string(layer) + "_attn_in" + dumpSuffix);
                V41DumpTensor(attnOut, "fl_layer" + std::to_string(layer) + "_attn_o" + dumpSuffix);
                V41DumpTensor(attnProj, "fl_layer" + std::to_string(layer) + "_attn" + dumpSuffix);
                if (cmpIdx != nullptr) {
                    V41DumpTensor(*cmpIdx, "fl_layer" + std::to_string(layer) + "_topk" + dumpSuffix);
                }
            }
            DeepSeekV4HcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            cache.totalLen += seqlen;
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + "_hidden_attn" + dumpSuffix);
            }

            // ---- FFN (MoE) ----
            V41HcMix(*curHidden, weight[pre + ".hc_ffn_fn"], weight[pre + ".hc_ffn_scale"],
                     weight[pre + ".hc_ffn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                     ffnPre, ffnPost, ffnComb);
            V41HcApplyPre(*curHidden, attnPre, x);
            V41RMSNormBF16(x, weight[pre + ".ffn_norm.weight"], rms_norm_eps, ffnInput);
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({seqlen, dim});

            // 路由：sqrt(softplus(logits / gate_temp))，bias 只参与选择
            {
                std::string gpre = pre + ".ffn.gate";
                Data xFloat, logits;
                ToDataType(ffnInput, xFloat, DataType::FLOAT32);
                Linear(xFloat, weight[gpre + ".weight"], Data(), logits);
                ToDataType(logits, DataType::FLOAT32);
                if (std::fabs(gate_temp - 1.0f) > 1e-6f) {
                    Mul(logits, 1.0f / gate_temp, logits);
                }
                Data &gateBias = weight[gpre + ".bias"];
                // 图像 token 使用 bias_vl 做专家选择（只影响 prefill，走 CPU 参考路径）
                Data *gateBiasVl = nullptr;
                if (hasImageTokens) {
                    auto vlIt = weight.weight.find(gpre + ".bias_vl");
                    AssertInFastLLM(vlIt != weight.weight.end(),
                                    "DeepSeekV41: " + gpre + ".bias_vl is required for image tokens.");
                    gateBiasVl = &vlIt->second;
                    gateBiasVl->ToDevice(DataDevice::CPU);
                }
                bool routed = false;
#ifdef USE_CUDA
                if (!hasImageTokens && logits.dataDevice == DataDevice::CUDA &&
                    !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE") &&
                    FastllmCudaDeepSeekV4RouteScoreTransform(logits, 2)) {
                    gateBias.ToDevice(DataDevice::CUDA);
                    SelectExpert(logits, expertIndex, expertScore, num_experts_per_tok, true,
                                 routed_scaling_factor, &gateBias);
                    routed = true;
                }
#endif
                if (!routed) {
                    logits.ToDevice(DataDevice::CPU);
                    gateBias.ToDevice(DataDevice::CPU);
                    const float *raw = (const float*)logits.cpuData;
                    const float *bias = (const float*)gateBias.cpuData;
                    std::vector<int> indices((uint64_t)seqlen * num_experts_per_tok);
                    std::vector<float> scores((uint64_t)seqlen * num_experts_per_tok);
                    std::vector<float> original(num_experts), select(num_experts);
                    for (int t = 0; t < seqlen; t++) {
                        const float *tokenBias = (gateBiasVl != nullptr && (*imageMask)[t] != 0) ?
                                                 (const float*)gateBiasVl->cpuData : bias;
                        for (int e = 0; e < num_experts; e++) {
                            original[e] = std::sqrt(V41Softplus(raw[(uint64_t)t * num_experts + e]));
                            select[e] = original[e] + tokenBias[e];
                        }
                        float sum = 0.0f;
                        for (int k = 0; k < num_experts_per_tok; k++) {
                            int best = 0;
                            for (int e = 1; e < num_experts; e++) {
                                if (select[e] > select[best]) {
                                    best = e;
                                }
                            }
                            indices[(uint64_t)t * num_experts_per_tok + k] = best;
                            scores[(uint64_t)t * num_experts_per_tok + k] = original[best];
                            sum += original[best];
                            select[best] = -std::numeric_limits<float>::infinity();
                        }
                        for (int k = 0; k < num_experts_per_tok; k++) {
                            float &v = scores[(uint64_t)t * num_experts_per_tok + k];
                            if (norm_topk_prob && num_experts_per_tok > 1) {
                                v /= (sum + 1e-20f);
                            }
                            v *= routed_scaling_factor;
                        }
                    }
                    Data idxData(DataType::INT32, {seqlen, num_experts_per_tok});
                    idxData.Allocate();
                    memcpy(idxData.cpuData, indices.data(), indices.size() * sizeof(int));
                    expertIndex.CopyFrom(idxData);
                    expertScore.CopyFrom(Data(DataType::FLOAT32, {seqlen, num_experts_per_tok}, scores));
                }
            }

            {
                std::vector<Data*> moeWeights = weights[layer];
                Data sharedExpertOut;
                bool hasSharedExpertOut = false;
                auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
                auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
                if (GetCudaSharedExpert() && sharedGateupIt != weight.weight.end() &&
                    sharedDownIt != weight.weight.end() && !sharedGateupIt->second.isDiskWeight &&
                    !sharedDownIt->second.isDiskWeight) {
                    Data ww1, ww3;
                    LinearSwigluBlock(&ffnInput, &sharedGateupIt->second, GetEmptyData(), &ww3, &ww1);
                    Linear(ww1, sharedDownIt->second, *GetEmptyData(), sharedExpertOut);
                    moeWeights[0] = moeWeights[1] = nullptr;
                    hasSharedExpertOut = true;
                }
                this->ApplyMoeDeviceMapForLayer(layer);
                MergeMOEBlock(&ffnInput, &expertIndex, &expertScore, &moeWeights, &biass[layer],
                              &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layer,
                              ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                              MoeGateSwiglu, false, swiglu_limit, true);
                ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
                if (hasSharedExpertOut) {
                    ffnOut.ToDevice(sharedExpertOut.dataDevice);
                    AddTo(ffnOut, sharedExpertOut);
                }
            }
            ffnOut.Reshape(ffnDims);
            if (dumpDebug) {
                V41DumpTensor(ffnInput, "fl_layer" + std::to_string(layer) + "_ffn_in" + dumpSuffix);
                V41DumpTensor(ffnOut, "fl_layer" + std::to_string(layer) + "_ffn" + dumpSuffix);
                V41DumpTensor(expertIndex, "fl_layer" + std::to_string(layer) + "_expert_idx" + dumpSuffix);
            }
            DeepSeekV4HcPost(ffnOut, *curHidden, ffnPost, ffnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            preMix.CopyFrom(ffnPre);
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + dumpSuffix);
            }
        }

        // ---- head（只取最后一个 token）----
        Data headInput;
        if (seqlen > 1) {
            Data lastHidden, lastPre;
            Split(*curHidden, 1, seqlen - 1, seqlen, lastHidden);
            Split(preMix, 1, seqlen - 1, seqlen, lastPre);
            V41HcApplyPre(lastHidden, lastPre, headInput);
        } else {
            V41HcApplyPre(*curHidden, preMix, headInput);
        }

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(1, 1);
        std::vector<GenerationConfig> generationConfigs(1, generationConfig);
        if (generationConfigs[0].do_sample && generationConfigs[0].top_k <= 1 &&
            generationConfigs[0].temperature > 1e-6f) {
            generationConfigs[0].top_k = 5;
        }
        std::vector<std::pair<Data*, Data*> > samplingPastKeyValues;
        for (auto &kvPair : pastKeyValues) {
            samplingPastKeyValues.push_back(std::make_pair(&kvPair.first, &kvPair.second));
        }
        LLMSamplingBlock(this, &headInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, 1, true, samplingSeqLens, samplingPastKeyValues,
                         generationConfigs, lastTokens, retLogits, ret);

        state->totalLen += seqlen;
        V41UpdateStubPastKeyValues(pastKeyValues, state->totalLen, block_cnt);
        return ret;
    }

    void DeepSeekV41Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41States.erase((const void*)&pastKeyValues);
        }
        this->kvCacheId = 0;
        elementsInKVCachePerToken = 0;
        for (int i = 0; i < block_cnt; i++) {
            if (pastKeyValues[i].first.dims.size() < 3) {
                continue;
            }
            elementsInKVCachePerToken +=
                (long long)pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                (long long)pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
        printf("finish.\n");
    }
}
