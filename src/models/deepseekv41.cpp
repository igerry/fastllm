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

        // BF16 KV 行 -> FP8 E4M3 + UE8M0（每 32 个一组）的 INT8 行，用于 FP8 KV 缓存存储
        void V41QuantizeKV(const Data &input, Data &output) {
            V41Executor().Run("DeepSeekV41QuantizeKV", {{"input", (Data*)&input}, {"output", &output}}, {}, {});
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
        this->canDoBatchForward = true;      // 多请求 decode 共享一次前向（见 ForwardSegments）
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
            // DSpark 草稿层（mtp.*，见 src/models/deepseekv41_dspark.cpp）
            "mtp.*.attn.wq_a.weight", "mtp.*.attn.wq_b.weight",
            "mtp.*.attn.wkv.weight",
            "mtp.*.attn.wo_a.weight", "mtp.*.attn.wo_b.weight",
            "mtp.*.ffn.gate.weight",
            "mtp.*.ffn.experts.*.w1.weight",
            "mtp.*.ffn.experts.*.w2.weight",
            "mtp.*.ffn.experts.*.w3.weight",
            "mtp.*.ffn.shared_experts.w1.weight",
            "mtp.*.ffn.shared_experts.w2.weight",
            "mtp.*.ffn.shared_experts.w3.weight",
            "mtp.*.main_proj.weight",
            "mtp.*.markov_head.head.weight",
            "mtp.*.confidence_head.proj.weight",
            "vision.patch_embed.proj.weight",
            "vision.blocks.*.attn.wqkv.weight", "vision.blocks.*.attn.wo.weight",
            "vision.blocks.*.mlp.w1.weight", "vision.blocks.*.mlp.w2.weight",
            "aligner.w1.weight", "aligner.w2.weight",
        };
    }

    DeepSeekV41Model::~DeepSeekV41Model() {
        ShutdownRuntime();
        DsparkReportStats();
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
        InitDsparkParams();

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
            // DSpark 草稿层：只有开启投机解码时才加载（默认跳过，省下约 30 GB 权重）
            if (V41StartsWith(name, "mtp.")) {
                if (!DsparkTensorNeeded(name)) {
                    continue;
                }
                if (V41EndsWith(name, ".confidence_head.proj.weight")) {
                    // 置信度是调度用的概率而不是激活，参考实现用 fp32 计算
                    result[name].push_back({name, DataType::FLOAT32});
                    continue;
                }
                if (V41EndsWith(name, ".markov_head.embed.weight")) {
                    result[name].push_back({name, DataType::BFLOAT16});
                    continue;
                }
                // 其余（attn_sink / gate.bias / gate.weight / 线性层）走下面的通用规则
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

    void DeepSeekV41Model::RunEngram(int layer, int engramLayerIndex, const std::vector<DeepSeekV41Segment> &segments,
                                     Data &hiddenStates) {
        AssertInFastLLM(engramMeta.loaded,
                        "DeepSeekV41: engram meta is not loaded. Generate engram_meta.json with "
                        "`python -m ftllm.deepseek_v41_engram <model_dir>` or set FASTLLM_DSV41_ENGRAM_META.");
        std::string pre = "layers." + std::to_string(layer) + ".engram";
        int total = 0;
        for (auto &seg : segments) {
            total += seg.seqlen;
        }
        const int cols = (engram_max_ngram_size - 1) * engram_n_heads;
        std::vector<int64_t> rows;
        rows.reserve((size_t)total * cols);
        std::vector<float> maskValues(total, 1.0f);
        bool hasDead = false;
        for (auto &seg : segments) {
            const std::vector<int> &history = seg.state->engramHistory;
            std::vector<int64_t> segRows;
            ComputeEngramHashes(engramLayerIndex, history, seg.startPos, seg.seqlen, segRows);
            rows.insert(rows.end(), segRows.begin(), segRows.end());
            for (int i = 0; i < seg.seqlen; i++) {
                if (history[seg.startPos + i] < 0) {
                    maskValues[seg.offset + i] = 0.0f;
                    hasDead = true;
                }
            }
        }
        Data gathered;
        GatherEngramRows(layer, rows, total, gathered);
        Data kv;
        Linear(gathered, weight[pre + ".wkv.weight"], Data(), kv);
        Data mask;
        if (hasDead) {
            mask.CopyFrom(Data(DataType::FLOAT32, {1, total}, maskValues));
        }
        V41EngramApply(hiddenStates, kv, weight[pre + ".q_weight"], weight[pre + ".k_weight"],
                       hasDead ? &mask : nullptr, rms_norm_eps);
    }

    // ==================== 请求状态 ====================

    namespace {
        // 原地清空请求状态（layers 里的 Data 没有深拷贝赋值，只能重建）
        bool V41HasImageToken(const std::vector<int> &tokens, int imageTokenId) {
            if (imageTokenId < 0) {
                return false;
            }
            for (int tok : tokens) {
                if (tok == imageTokenId) {
                    return true;
                }
            }
            return false;
        }

        void V41ResetState(DeepSeekV41RequestState &state, int blockCnt) {
            state.layers.clear();
            state.layers.resize(blockCnt);
            state.engramHistory.clear();
            state.totalLen = 0;
            state.restoredLen = 0;
            // pendingMultimodal 由 ResponseContext 持有，重建时保留；图像嵌入需要重新编码
            state.imagesEncoded = false;
            state.imageSpans.clear();
            // DSpark：滑窗与待发队列都是相对旧缓存的，一并丢弃
            state.dspark.reset();
        }
    }

    void DeepSeekV41Model::RegisterState(const void *vectorKey, const void *firstKey,
                                         const std::shared_ptr<DeepSeekV41RequestState> &state) {
        if (vectorKey != nullptr) {
            v41States[vectorKey] = state;
        }
        if (firstKey != nullptr) {
            v41StatesByFirstKey[firstKey] = state;
        }
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::GetOrCreateState(
            std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset) {
        const void *key = (const void*)&pastKeyValues;
        const void *firstKey = pastKeyValues.empty() ? nullptr : (const void*)&pastKeyValues[0].first;
        std::lock_guard<std::mutex> guard(v41StateMutex);
        auto it = v41States.find(key);
        if (it != v41States.end()) {
            if (reset) {
                V41ResetState(*it->second, block_cnt);
            }
            if (firstKey != nullptr) {
                v41StatesByFirstKey[firstKey] = it->second;
            }
            return it->second;
        }
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(block_cnt);
        RegisterState(key, firstKey, state);
        return state;
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::GetStateByFirstKey(const Data *firstKey) {
        std::lock_guard<std::mutex> guard(v41StateMutex);
        auto it = v41StatesByFirstKey.find((const void*)firstKey);
        return it == v41StatesByFirstKey.end() ? nullptr : it->second;
    }

    void DeepSeekV41Model::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        const void *firstKey = context->pastKeyValues.empty() ? nullptr : (const void*)&context->pastKeyValues[0].first;
        std::lock_guard<std::mutex> guard(v41StateMutex);
        std::shared_ptr<DeepSeekV41RequestState> state;
        if (v41PendingRestoredState) {
            state = v41PendingRestoredState;
            v41PendingRestoredState.reset();
        } else {
            auto existing = v41States.find(key);
            if (existing != v41States.end()) {
                state = existing->second;
            } else {
                state = std::make_shared<DeepSeekV41RequestState>();
                state->layers.resize(block_cnt);
            }
        }
        RegisterState(key, firstKey, state);
        // 图文请求：调度器可能只用普通 Forward 逐块 prefill，因此在这里就把多模态输入记到请求状态里，
        // 由第一个 prefill 块编码图像（见 ForwardSegments 的调用方）
        if (!context->multimodalInput.empty()) {
            state->pendingMultimodal = &context->multimodalInput;
        }
    }

    void DeepSeekV41Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(v41StateMutex);
        v41States.erase((const void*)&context->pastKeyValues);
        if (!context->pastKeyValues.empty()) {
            v41StatesByFirstKey.erase((const void*)&context->pastKeyValues[0].first);
        }
    }

    // ==================== 前缀缓存 ====================

    namespace {
        bool V41PrefixCacheDisabled() {
            static const bool disabled = V41EnvFlag("FASTLLM_DSV41_DISABLE_PREFIX_CACHE");
            return disabled;
        }

        bool V41PrefixCacheDebug() {
            static const bool debug = V41EnvFlag("FASTLLM_DSV41_PREFIX_CACHE_DEBUG");
            return debug;
        }

        int V41EnvInt(const char *name, int fallback) {
            const char *v = std::getenv(name);
            if (v == nullptr || v[0] == '\0') {
                return fallback;
            }
            return atoi(v);
        }

        // 快照：深拷贝到 CPU（保留 expansion 容量，restore 后可继续追加）
        void V41SnapshotTensor(Data &dst, const Data &src) {
            if (src.dims.size() == 0 || src.Count(0) == 0) {
                return;
            }
            dst.CopyFrom(src);
            dst.ToDevice(DataDevice::CPU);
        }

        // 恢复：CPU -> CPU 深拷贝，随后由执行器在首次使用时搬到计算设备
        void V41RestoreTensor(Data &dst, const Data &src) {
            if (src.dims.size() == 0 || src.Count(0) == 0) {
                return;
            }
            dst.CopyFrom(src);
            dst.SetKVCache();
        }
    }

    void DeepSeekV41HistoryCacheManager::Record(const std::shared_ptr<DeepSeekV41HistoryMemory> &memory) {
        if (!memory || memory->totalLen <= 0 || (int)memory->tokens.size() != memory->totalLen) {
            return;
        }
        std::lock_guard<std::mutex> guard(this->locker);
        int commonMax = V41EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", this->maxRecordNum);
        this->maxRecordNum = std::max(1, V41EnvInt("FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS", commonMax));
        auto old = this->memorys.find(memory->tokens);
        if (old != this->memorys.end()) {
            memory->recordTimes = old->second->recordTimes + 1;
            memory->flushTime = ++this->flushTime;
            old->second = memory;
            return;
        }
        while ((int)this->memorys.size() >= this->maxRecordNum) {
            auto eraseIt = this->memorys.end();
            long long minFlushTime = (1LL << 60);
            for (auto it = this->memorys.begin(); it != this->memorys.end(); ++it) {
                if (it->second->flushTime < minFlushTime) {
                    minFlushTime = it->second->flushTime;
                    eraseIt = it;
                }
            }
            if (eraseIt == this->memorys.end()) {
                break;
            }
            this->memorys.erase(eraseIt);
        }
        memory->recordTimes = 1;
        memory->flushTime = ++this->flushTime;
        this->memorys[memory->tokens] = memory;
    }

    std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> >
    DeepSeekV41HistoryCacheManager::GetCandidates(const std::vector<int> &inputTokens) {
        std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> > candidates;
        std::lock_guard<std::mutex> guard(this->locker);
        // 至少留一个 token 给本次前向
        const int maxLen = (int)inputTokens.size() - 1;
        for (auto &it : this->memorys) {
            const std::vector<int> &tokens = it.first;
            int limit = std::min(maxLen, (int)tokens.size());
            int len = 0;
            while (len < limit && tokens[len] == inputTokens[len]) {
                len++;
            }
            if (len > 0) {
                candidates.push_back({it.second, len});
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> &a,
                            const std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> &b) {
                             if (a.second != b.second) {
                                 return a.second > b.second;
                             }
                             return a.first->totalLen < b.first->totalLen;
                         });
        return candidates;
    }

    bool DeepSeekV41Model::CanTruncateHistory(const DeepSeekV41HistoryMemory &memory, int len) const {
        const int total = memory.totalLen;
        if (len <= 0 || len > total) {
            return false;
        }
        // 滑窗环形缓存只保留最后 window_size 个位置；截断到 len 后需要 [len - W + 1, len) 仍然完整
        if (std::max(0, len - window_size + 1) < std::max(0, total - window_size)) {
            return false;
        }
        // 压缩 KV：凑不满一组的原始尾块只在 len == total 时可用
        for (int layer : kv_source_layer_ids) {
            int ratio = compress_ratios[layer];
            if (ratio > 1 && len % ratio != 0 && len != total) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<DeepSeekV41HistoryMemory> DeepSeekV41Model::SnapshotState(
            const DeepSeekV41RequestState &state, const std::vector<int> &allTokens) {
        const int totalLen = state.totalLen;
        if (totalLen <= 0 || (int)allTokens.size() < totalLen || (int)state.layers.size() != block_cnt) {
            return nullptr;
        }
        if (!engram_layer_ids.empty() && (int)state.engramHistory.size() < totalLen) {
            return nullptr;
        }
        auto memory = std::make_shared<DeepSeekV41HistoryMemory>();
        memory->totalLen = totalLen;
        memory->tokens.assign(allTokens.begin(), allTokens.begin() + totalLen);
        memory->engramHistory.assign(state.engramHistory.begin(),
                                     state.engramHistory.begin() + std::min((int)state.engramHistory.size(), totalLen));
        memory->layers.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            const DeepSeekV41LayerCache &src = state.layers[layer];
            DeepSeekV41LayerCache &dst = memory->layers[layer];
            if (src.totalLen != totalLen) {
                return nullptr;
            }
            dst.totalLen = src.totalLen;
            dst.compressedBlocks = src.compressedBlocks;
            dst.rawTail = src.rawTail;
            V41SnapshotTensor(dst.windowKV, src.windowKV);
            if (isKvSource[layer]) {
                V41SnapshotTensor(dst.compressedKV, src.compressedKV);
                V41SnapshotTensor(dst.indexK, src.indexK);
                if (src.rawTail > 0) {
                    V41SnapshotTensor(dst.rawTailKV, src.rawTailKV);
                    V41SnapshotTensor(dst.rawTailScore, src.rawTailScore);
                }
            }
        }
        return memory;
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::RestoreState(
            const DeepSeekV41HistoryMemory &memory, int len) {
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(block_cnt);
        state->totalLen = len;
        state->restoredLen = len;
        if (!engram_layer_ids.empty()) {
            state->engramHistory.assign(memory.engramHistory.begin(), memory.engramHistory.begin() + len);
        }
        const bool exact = len == memory.totalLen;
        for (int layer = 0; layer < block_cnt; layer++) {
            const DeepSeekV41LayerCache &src = memory.layers[layer];
            DeepSeekV41LayerCache &dst = state->layers[layer];
            dst.totalLen = len;
            // 环形缓存整体恢复；位置 >= len 的行不会被读取，之后会被新 token 覆盖
            V41RestoreTensor(dst.windowKV, src.windowKV);
            if (!isKvSource[layer]) {
                continue;
            }
            const int ratio = compress_ratios[layer];
            const int blocks = len / ratio;
            dst.compressedBlocks = blocks;
            if (blocks > 0) {
                V41RestoreTensor(dst.compressedKV, src.compressedKV);
                if (dst.compressedKV.dims.size() == 3 && dst.compressedKV.dims[1] > blocks) {
                    dst.compressedKV.Resize({dst.compressedKV.dims[0], blocks, dst.compressedKV.dims[2]});
                }
                if (isIndexSource[layer]) {
                    V41RestoreTensor(dst.indexK, src.indexK);
                    if (dst.indexK.dims.size() == 3 && dst.indexK.dims[1] > blocks) {
                        dst.indexK.Resize({dst.indexK.dims[0], blocks, dst.indexK.dims[2]});
                    }
                }
            }
            if (exact && src.rawTail > 0) {
                dst.rawTail = src.rawTail;
                V41RestoreTensor(dst.rawTailKV, src.rawTailKV);
                V41RestoreTensor(dst.rawTailScore, src.rawTailScore);
            } else {
                dst.rawTail = 0;
            }
        }
        return state;
    }

    void DeepSeekV41Model::TryRecordResponseContext(ResponseContext *context) {
        if (context == nullptr || !this->saveHistoryChat || V41PrefixCacheDisabled()) {
            return;
        }
        std::shared_ptr<DeepSeekV41RequestState> state;
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            auto it = v41States.find((const void*)&context->pastKeyValues);
            if (it != v41States.end()) {
                state = it->second;
            }
        }
        if (!state || state->totalLen <= 0 || context->allTokens.empty()) {
            return;
        }
        // 图文请求的状态不进前缀缓存（原因同 TryRestoreHistoryCache）
        if (!context->multimodalInput.empty() || V41HasImageToken(context->allTokens, image_token_id)) {
            return;
        }
        auto memory = SnapshotState(*state, context->allTokens);
        if (!memory) {
            if (V41PrefixCacheDebug()) {
                printf("[fastllm-dsv41-prefix-cache] skip record: total_len=%d all_tokens=%d\n",
                       state->totalLen, (int)context->allTokens.size());
                fflush(stdout);
            }
            return;
        }
        v41HistoryCache.Record(memory);
        if (V41PrefixCacheDebug()) {
            printf("[fastllm-dsv41-prefix-cache] record tokens=%d records=%d\n",
                   memory->totalLen, (int)v41HistoryCache.memorys.size());
            fflush(stdout);
        }
    }

    bool DeepSeekV41Model::TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) {
        cacheLen = 0;
        if (!this->saveHistoryChat || V41PrefixCacheDisabled()) {
            return false;
        }
        const int minTokens = std::max(1, V41EnvInt("FASTLLM_DSV41_PREFIX_CACHE_MIN_TOKENS", 16));
        if ((int)inputTokens.size() <= minTokens) {
            return false;
        }
        // 图文请求不复用前缀：图像占位 token 的 id 与图像内容无关，同样的文字配不同的图会误命中
        if (V41HasImageToken(inputTokens, image_token_id)) {
            return false;
        }
        auto candidates = v41HistoryCache.GetCandidates(inputTokens);
        std::shared_ptr<DeepSeekV41HistoryMemory> memory;
        int hitLen = 0, len = 0;
        for (auto &candidate : candidates) {
            if (candidate.second < minTokens) {
                break;
            }
            // 截断约束（滑窗 / 压缩尾块）最多需要回退几个 token
            int cur = candidate.second;
            for (int step = 0; step < 4 && cur >= minTokens && !CanTruncateHistory(*candidate.first, cur); step++) {
                cur--;
            }
            if (cur >= minTokens && CanTruncateHistory(*candidate.first, cur)) {
                memory = candidate.first;
                hitLen = candidate.second;
                len = cur;
                break;
            }
        }
        if (!memory) {
            if (V41PrefixCacheDebug()) {
                printf("[fastllm-dsv41-prefix-cache] miss input_tokens=%d candidates=%d best_lcp=%d\n",
                       (int)inputTokens.size(), (int)candidates.size(),
                       candidates.empty() ? 0 : candidates[0].second);
                fflush(stdout);
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> guard(v41HistoryCache.locker);
            memory->flushTime = ++v41HistoryCache.flushTime;
        }
        auto state = RestoreState(*memory, len);
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41PendingRestoredState = state;
        }
        inputTokens.erase(inputTokens.begin(), inputTokens.begin() + len);
        cacheLen = len;
        if (V41PrefixCacheDebug()) {
            printf("[fastllm-dsv41-prefix-cache] hit len=%d (lcp=%d record=%d) remaining=%d\n",
                   len, hitLen, memory->totalLen, (int)inputTokens.size());
            fflush(stdout);
        }
        return true;
    }

    void DeepSeekV41Model::TryRecordHistoryCache(const std::vector<int> &allTokens) {
        // 状态与 ResponseContext 绑定，记录在 TryRecordResponseContext 中完成
        (void)allTokens;
    }

    // ==================== 前向 ====================

    namespace {
        // 只更新调度器读取的第 0 层占位 KV（kvCacheId == 0）
        void V41UpdateStubKV(Data &key, Data &value, int totalLen) {
            int paddedLen = (std::max(totalLen, 1) / 128 + 1) * 128;
            std::vector<float> zeros((uint64_t)totalLen, 0.0f);
            Data stubKey(DataType::FLOAT32, {1, totalLen, 1}, zeros);
            Data stubValue(DataType::FLOAT32, {1, totalLen, 1}, zeros);
            stubKey.SetKVCache();
            stubValue.SetKVCache();
            stubKey.Expansion({1, paddedLen, 1});
            stubValue.Expansion({1, paddedLen, 1});
            key.FreeSpace();
            value.FreeSpace();
            key = Data();
            value = Data();
            key.CopyFrom(stubKey);
            value.CopyFrom(stubValue);
            key.SetKVCache();
            value.SetKVCache();
        }

        int V41FirstPosition(const Data *positionIds) {
            if (positionIds == nullptr || positionIds->dims.size() == 0 || positionIds->Count(0) == 0) {
                return 0;
            }
            auto pids = V41ReadTokenIds(*positionIds);
            return pids.empty() ? 0 : pids[0];
        }
    }

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
        const int startPos = V41FirstPosition(&positionIds);
        AssertInFastLLM(inputEmbeds == nullptr ||
                        (inputEmbeds->dims.size() == 3 && inputEmbeds->dims[1] == seqlen &&
                         inputEmbeds->dims[2] == embed_dim),
                        "DeepSeekV41Model: inputEmbeds must be [1, seqlen, dim].");
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model: imageMask length mismatch.");
        // 重建缓存时 V41ResetState 保留 pendingMultimodal（图像会重新编码）
        auto state = GetOrCreateState(pastKeyValues, startPos == 0);

        // ---- DSpark：已经校验通过的 token 直接出队，不需要前向 ----
        if (v41DsparkEnabled && state->dspark && !state->dspark->pending.empty()) {
            int queued = DsparkTakePending(*state, inputIds, seqlen);
            if (queued >= 0) {
                if (!pastKeyValues.empty()) {
                    V41UpdateStubKV(pastKeyValues[0].first, pastKeyValues[0].second, state->totalLen);
                }
                return std::vector<int>{queued};
            }
        }
        AssertInFastLLM(state->totalLen == startPos,
                        "DeepSeekV41Model: position mismatch (cache has " + std::to_string(state->totalLen) +
                        " tokens, request starts at " + std::to_string(startPos) + ").");

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

        std::vector<DeepSeekV41Segment> segments(1);
        segments[0].state = state;
        segments[0].startPos = startPos;
        segments[0].seqlen = seqlen;
        segments[0].offset = 0;
        std::vector<GenerationConfig> generationConfigs(1, generationConfig);
        std::vector<std::pair<Data*, Data*> > samplingPastKeyValues;
        for (auto &kvPair : pastKeyValues) {
            samplingPastKeyValues.push_back(std::make_pair(&kvPair.first, &kvPair.second));
        }
        std::vector<int> ret = ForwardSegmentsWithDspark(segments, inputIds, inputEmbeds, imageMask,
                                                         generationConfigs, lastTokens, retLogits,
                                                         samplingPastKeyValues, true);
        if (!pastKeyValues.empty()) {
            V41UpdateStubKV(pastKeyValues[0].first, pastKeyValues[0].second, state->totalLen);
        }
        return ret;
    }

    std::vector<int> DeepSeekV41Model::ForwardBatch(int batch, const Data &inputIds,
                                                    const std::vector<Data*> &attentionMask,
                                                    const std::vector<Data*> &positionIds,
                                                    const std::vector<int> &seqLens,
                                                    std::vector<std::pair<Data*, Data*> > &pastKeyValues,
                                                    const std::vector<GenerationConfig> &generationConfigs,
                                                    const LastTokensManager &lastTokens,
                                                    std::vector<std::vector<float>*> *retLogits) {
        (void)attentionMask;
        AssertInFastLLM(batch >= 1 && (int)seqLens.size() == batch && (int)positionIds.size() == batch &&
                        (int)pastKeyValues.size() == batch * block_cnt &&
                        (int)generationConfigs.size() == batch,
                        "DeepSeekV41Model::ForwardBatch: inconsistent batch arguments.");
        int total = 0;
        for (int len : seqLens) {
            total += len;
        }
        AssertInFastLLM(inputIds.Count(0) == (uint64_t)total,
                        "DeepSeekV41Model::ForwardBatch: inputIds length does not match seqLens.");

        // ---- DSpark：已经校验通过的 token 直接出队，这些请求不参与本次前向 ----
        // （批量前向里不做投机，但仍然要采集 main hidden，让草稿侧的滑窗跟上目标缓存）
        std::vector<int> pendingRet(batch, -1);
        std::vector<char> isPending(batch, 0);
        int pendingCount = 0;
        if (v41DsparkEnabled) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            int scan = 0;
            for (int i = 0; i < batch; i++) {
                auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                if (state && state->dspark && !state->dspark->pending.empty()) {
                    Data one;
                    Split(cpuIds, cpuIds.dims.size() - 1, scan, scan + seqLens[i], one);
                    int queued = DsparkTakePending(*state, one, seqLens[i]);
                    if (queued >= 0) {
                        pendingRet[i] = queued;
                        isPending[i] = 1;
                        pendingCount++;
                    }
                }
                scan += seqLens[i];
            }
        }
        if (pendingCount == batch) {
            for (int i = 0; i < batch; i++) {
                auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                if (state) {
                    V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first,
                                    *pastKeyValues[(size_t)i * block_cnt].second, state->totalLen);
                }
            }
            return pendingRet;
        }
        // 有请求出队时，把剩下的请求重新拼成一次前向
        if (pendingCount > 0) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            std::vector<Data*> activeMask, activePosition;
            std::vector<int> activeSeqLens, activeIndex;
            std::vector<std::pair<Data*, Data*> > activePast;
            std::vector<GenerationConfig> activeConfigs;
            std::vector<std::vector<float>*> activeLogits;
            LastTokensManager activeTokens;
            Data activeIds, part, catTmp;
            int scan = 0;
            bool first = true;
            for (int i = 0; i < batch; i++) {
                if (!isPending[i]) {
                    Split(cpuIds, cpuIds.dims.size() - 1, scan, scan + seqLens[i], part);
                    if (first) {
                        activeIds.CopyFrom(part);
                        first = false;
                    } else {
                        Cat(activeIds, part, activeIds.dims.size() - 1, catTmp);
                        activeIds.CopyFrom(catTmp);
                    }
                    activePosition.push_back(positionIds[i]);
                    activeSeqLens.push_back(seqLens[i]);
                    activeConfigs.push_back(generationConfigs[i]);
                    activeIndex.push_back(i);
                    for (int l = 0; l < block_cnt; l++) {
                        activePast.push_back(pastKeyValues[(size_t)i * block_cnt + l]);
                    }
                    if ((int)lastTokens.units.size() > i) {
                        activeTokens.units.push_back(lastTokens.units[i]);
                    }
                    if (retLogits != nullptr && (int)retLogits->size() > i) {
                        activeLogits.push_back((*retLogits)[i]);
                    }
                }
                scan += seqLens[i];
            }
            std::vector<Data*> emptyMask;
            auto sub = ForwardBatch((int)activeSeqLens.size(), activeIds, emptyMask, activePosition,
                                    activeSeqLens, activePast, activeConfigs, activeTokens,
                                    retLogits != nullptr ? &activeLogits : nullptr);
            for (size_t k = 0; k < activeIndex.size(); k++) {
                pendingRet[activeIndex[k]] = sub[k];
            }
            for (int i = 0; i < batch; i++) {
                if (isPending[i]) {
                    auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                    if (state) {
                        V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first,
                                        *pastKeyValues[(size_t)i * block_cnt].second, state->totalLen);
                    }
                }
            }
            return pendingRet;
        }

        std::vector<DeepSeekV41Segment> segments(batch);
        int offset = 0;
        for (int i = 0; i < batch; i++) {
            const int startPos = V41FirstPosition(positionIds[i]);
            auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
            if (!state) {
                AssertInFastLLM(startPos == 0,
                                "DeepSeekV41Model::ForwardBatch: request state is missing for a continuing request.");
                state = std::make_shared<DeepSeekV41RequestState>();
                state->layers.resize(block_cnt);
                std::lock_guard<std::mutex> guard(v41StateMutex);
                RegisterState(nullptr, (const void*)pastKeyValues[(size_t)i * block_cnt].first, state);
            } else if (startPos == 0 && state->totalLen != 0) {
                V41ResetState(*state, block_cnt);
            }
            AssertInFastLLM(state->totalLen == startPos,
                            "DeepSeekV41Model: position mismatch in batch (cache has " +
                            std::to_string(state->totalLen) + " tokens, request starts at " +
                            std::to_string(startPos) + ").");
            segments[i].state = state;
            segments[i].startPos = startPos;
            segments[i].seqlen = seqLens[i];
            segments[i].offset = offset;
            offset += seqLens[i];
        }

        std::vector<int> ret;
        if (batch > 1) {
            static bool announced = false;
            if (!announced) {
                announced = true;
                printf("[Fastllm] DeepSeek-V4.1: batched forward active (batch = %d).\n", batch);
                fflush(stdout);
            }
        }
        // 含长 prefill 片段的混合批次退回为逐个前向，避免一次前向的激活内存过大
        bool splitBatch = false;
        if (batch > 1) {
            const int chunkLimit = std::max(1, GetChunkedPrefillSize());
            for (auto &seg : segments) {
                if (seg.seqlen > chunkLimit) {
                    splitBatch = true;
                }
            }
        }
        if (splitBatch) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            for (int i = 0; i < batch; i++) {
                std::vector<DeepSeekV41Segment> one(1);
                one[0] = segments[i];
                one[0].offset = 0;
                Data curIds;
                Split(cpuIds, 1, segments[i].offset, segments[i].offset + segments[i].seqlen, curIds);
                std::vector<GenerationConfig> curConfigs(1, generationConfigs[i]);
                LastTokensManager curTokens;
                if ((int)lastTokens.units.size() > i) {
                    curTokens.units.push_back(lastTokens.units[i]);
                }
                std::vector<std::vector<float>*> curLogits;
                if (retLogits != nullptr && (int)retLogits->size() > i) {
                    curLogits.push_back((*retLogits)[i]);
                }
                std::vector<std::pair<Data*, Data*> > curPastKeyValues(
                        pastKeyValues.begin() + (size_t)i * block_cnt,
                        pastKeyValues.begin() + (size_t)(i + 1) * block_cnt);
                auto cur = ForwardSegmentsWithDspark(one, curIds, nullptr, nullptr, curConfigs, curTokens,
                                                     retLogits != nullptr ? &curLogits : nullptr,
                                                     curPastKeyValues, true);
                ret.push_back(cur[0]);
            }
        } else {
            ret = ForwardSegmentsWithDspark(segments, inputIds, nullptr, nullptr, generationConfigs, lastTokens,
                                            retLogits, pastKeyValues, batch == 1);
        }
        for (int i = 0; i < batch; i++) {
            V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first, *pastKeyValues[(size_t)i * block_cnt].second,
                            segments[i].state->totalLen);
        }
        return ret;
    }

    std::vector<int> DeepSeekV41Model::ForwardSegments(std::vector<DeepSeekV41Segment> &segments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues) {
        const int numSegments = (int)segments.size();
        AssertInFastLLM(numSegments >= 1 && (int)generationConfigsIn.size() == numSegments,
                        "DeepSeekV41Model::ForwardSegments: bad segments.");
        int total = 0;
        for (auto &seg : segments) {
            AssertInFastLLM(seg.state && seg.seqlen > 0 && seg.offset == total && seg.state->totalLen == seg.startPos,
                            "DeepSeekV41Model::ForwardSegments: inconsistent segment.");
            total += seg.seqlen;
        }
        const bool single = numSegments == 1;
        const int seqlen = total;   // 拼接后的 token 总数
        // 图像 token（掩码按拼接后的全局下标）：Engram 置 -1，专家选择改用 gate.bias_vl
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model::ForwardSegments: imageMask length mismatch.");
        bool hasImageTokens = false;
        if (imageMask != nullptr) {
            for (int v : *imageMask) {
                if (v != 0) {
                    hasImageTokens = true;
                    break;
                }
            }
        }
        // --kv_cache_dtype fp8_e4m3：滑窗 KV 与压缩 KV 以 FP8 + UE8M0 块 scale 存储（默认 BF16）
        const bool fp8KV = this->kvCacheDataType == DataType::FP8_E4M3;

        // ---- Engram 历史 ----
        std::vector<int> tokenIds = V41ReadTokenIds(inputIds);
        AssertInFastLLM((int)tokenIds.size() == total, "DeepSeekV41Model::ForwardSegments: inputIds length mismatch.");
        if (!engram_layer_ids.empty()) {
            for (auto &seg : segments) {
                AssertInFastLLM((int)seg.state->engramHistory.size() == seg.startPos,
                                "DeepSeekV41: engram history is out of sync with the cache.");
                for (int i = 0; i < seg.seqlen; i++) {
                    int tok = tokenIds[seg.offset + i];
                    int compressed = -1;
                    bool isImage = tok == image_token_id ||
                                   (imageMask != nullptr && (int)imageMask->size() > seg.offset + i &&
                                    (*imageMask)[seg.offset + i] != 0);
                    if (!isImage && engramMeta.loaded && tok >= 0 && tok < (int)engramMeta.tokenMap.size()) {
                        compressed = engramMeta.tokenMap[tok];
                    }
                    seg.state->engramHistory.push_back(compressed);
                }
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
                AssertInFastLLM(inputEmbeds->Count(0) == (uint64_t)seqlen * dim,
                                "DeepSeekV41Model::ForwardSegments: inputEmbeds shape mismatch.");
                embedOut.CopyFrom(*inputEmbeds);
                ToDataType(embedOut, DataType::BFLOAT16);
            } else {
                Embedding(inputIds, weight["embed.weight"], embedOut);
                ToDataType(embedOut, DataType::BFLOAT16);
            }
            embedOut.Reshape({1, seqlen, 1, dim});
            Repeat(embedOut, 2, hc_mult, hiddenStates);
        }
        Data *curHidden = &hiddenStates;
        Data *nextHidden = &hiddenTemp;
        const bool dumpDebug = single && std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr;
        const int startPos0 = segments[0].startPos;
        const std::string dumpSuffix = startPos0 == 0 ? std::string("") : "_p" + std::to_string(startPos0);
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

        // 片段切片：单片段时直接使用整批张量，多片段时按 offset 拷贝出来
        auto sliceOf = [&](Data &full, int i, Data &tmp) -> Data* {
            if (single) {
                return &full;
            }
            Split(full, 1, segments[i].offset, segments[i].offset + segments[i].seqlen, tmp);
            return &tmp;
        };
        // 按 axis=1 拼接多个片段（两块临时缓冲交替使用，避免输出与输入别名）
        auto catSegments = [&](std::vector<Data> &parts, Data *tmp) -> Data* {
            Data *acc = &parts[0];
            int t = 0;
            for (size_t i = 1; i < parts.size(); i++) {
                Cat(*acc, parts[i], 1, tmp[t]);
                acc = &tmp[t];
                t ^= 1;
            }
            return acc;
        };

        // 本次前向内跨层共享的 indexer 结果（按片段）
        std::vector<Data> segTopK(numSegments), segCandidate(numSegments);
        std::vector<char> segHasCandidates(numSegments, 0);

        Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
        Data x, attnInput, qr, qNorm, q, kv, attnOut, woAOut, attnProj;
        Data ffnInput, ffnOut, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        std::vector<Data> segQ(numSegments), segKV(numSegments), segAttnOut(numSegments);
        Data catTmp[2];

        for (int layer = 0; layer < block_cnt; layer++) {
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            std::string pre = "layers." + std::to_string(layer);
            const int ratio = compress_ratios[layer];
            const V41RopeParams &rope = ratio > 0 ? compressRope : windowRope;

            // ---- Engram ----
            for (size_t l = 0; l < engram_layer_ids.size(); l++) {
                if (engram_layer_ids[l] == layer) {
                    RunEngram(layer, (int)l, segments, *curHidden);
                }
            }

            // ---- DSpark：目标层的 main hidden（attention 的输入，对 hc 份取均值）----
            if (v41DsparkEnabled && (int)v41IsDsparkTarget.size() == block_cnt && v41IsDsparkTarget[layer]) {
                int slot = 0;
                for (size_t k = 0; k < v41DsparkTargetLayerIds.size(); k++) {
                    if (v41DsparkTargetLayerIds[k] == layer) {
                        slot = (int)k;
                        break;
                    }
                }
                for (int s = 0; s < numSegments; s++) {
                    DeepSeekV41SpecScratch *spec = segments[s].spec;
                    if (spec == nullptr || !spec->captureMain) {
                        continue;
                    }
                    if (spec->mainHidden.size() != v41DsparkTargetLayerIds.size()) {
                        spec->mainHidden.resize(v41DsparkTargetLayerIds.size());
                    }
                    Data segHidden;
                    Data *src = sliceOf(*curHidden, s, segHidden);
                    Data uniform;
                    std::vector<float> uniformValues((uint64_t)segments[s].seqlen * hc_mult,
                                                     1.0f / (float)hc_mult);
                    uniform.CopyFrom(Data(DataType::FLOAT32, {1, segments[s].seqlen, hc_mult}, uniformValues));
                    V41HcApplyPre(*src, uniform, spec->mainHidden[slot]);
                }
            }

            // ---- attention（整批部分）----
            V41HcMix(*curHidden, weight[pre + ".hc_attn_fn"], weight[pre + ".hc_attn_scale"],
                     weight[pre + ".hc_attn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                     attnPre, attnPost, attnComb);
            V41HcApplyPre(*curHidden, preMix, x);
            V41RMSNormBF16(x, weight[pre + ".attn_norm.weight"], rms_norm_eps, attnInput);

            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr);
            V41RMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({1, seqlen, num_attention_heads, headDim});

            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv);
            V41RMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            kv.Reshape({1, seqlen, headDim});

            // kv source 层：compressor 的投影按整批算，分组 / 追加按片段做
            Data rawKVAll, rawScoreAll;
            // index source 层：indexer 的 q / 权重投影按整批算
            Data qIdxAll, idxWeightsAll;
            bool needIndexer = false;
            if (ratio > 0 && isKvSource[layer]) {
                std::string cpre = pre + ".attn.compressor";
                Data xFloat;
                ToDataType(attnInput, xFloat, DataType::FLOAT32);
                Linear(xFloat, weight[cpre + ".wkv.weight"], Data(), rawKVAll);
                ToDataType(rawKVAll, DataType::FLOAT32);
                if (ratio > 1) {
                    Linear(xFloat, weight[cpre + ".wgate.weight"], Data(), rawScoreAll);
                    ToDataType(rawScoreAll, DataType::FLOAT32);
                }
            }
            if (ratio > 0 && isIndexSource[layer]) {
                std::string ipre = pre + ".attn.indexer";
                Linear(qNorm, weight[ipre + ".wq_b.weight"], Data(), qIdxAll);
                qIdxAll.Reshape({1, seqlen, index_n_heads, index_head_dim});
                Data idxWeights;
                Linear(attnInput, weight[ipre + ".weights_proj.weight"], Data(), idxWeights);
                ToDataType(idxWeights, DataType::FLOAT32);
                Mul(idxWeights, (1.0f / std::sqrt((float)index_head_dim)) * (1.0f / std::sqrt((float)index_n_heads)),
                    idxWeightsAll);
                needIndexer = true;
            }

            // ---- attention（按片段）----
            for (int s = 0; s < numSegments; s++) {
                DeepSeekV41Segment &seg = segments[s];
                DeepSeekV41LayerCache &cache = seg.state->layers[layer];
                const int startPos = seg.startPos;
                const int segLen = seg.seqlen;

                Data *qSeg = sliceOf(q, s, segQ[s]);
                V41RotaryQuant(*qSeg, rope, startPos, 1, false, 0, 32);
                Data *kvSeg = sliceOf(kv, s, segKV[s]);
                V41RotaryQuant(*kvSeg, rope, startPos, 1, false, 1, 32);

                Data *compressedKV = nullptr;
                Data *cmpIdx = nullptr;
                if (ratio > 0) {
                    const int src = kvSourceOf[layer];
                    DeepSeekV41LayerCache &srcCache = seg.state->layers[src];
                    if (isKvSource[layer]) {
                        std::string cpre = pre + ".attn.compressor";
                        Data rawKVSeg, rawScoreSeg, allKV, allScore;
                        Data *rawKV = sliceOf(rawKVAll, s, rawKVSeg);
                        Data *rawScore = ratio > 1 ? sliceOf(rawScoreAll, s, rawScoreSeg) : nullptr;
                        Data *kvAll = rawKV, *scoreAll = rawScore;
                        const int specPrevTail = cache.rawTail;
                        const int specPrevBlocks = cache.compressedBlocks;
                        if (cache.rawTail > 0) {
                            Cat(cache.rawTailKV, *rawKV, 1, allKV);
                            kvAll = &allKV;
                            if (ratio > 1) {
                                Cat(cache.rawTailScore, *rawScore, 1, allScore);
                                scoreAll = &allScore;
                            }
                        }
                        // DSpark 校验：保存压缩器的原始输入流与回滚点，接受长度确定后重建
                        if (seg.spec != nullptr && seg.spec->deferWindow) {
                            seg.spec->prevRawTail[layer] = specPrevTail;
                            seg.spec->prevBlocks[layer] = specPrevBlocks;
                            seg.spec->rawKV[layer].CopyFrom(*kvAll);
                            if (ratio > 1) {
                                seg.spec->rawScore[layer].CopyFrom(*scoreAll);
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
                            if (fp8KV) {
                                Data latent8;
                                V41QuantizeKV(latent, latent8);
                                V41AppendRows(cache.compressedKV, latent8);
                            } else {
                                V41AppendRows(cache.compressedKV, latent);
                            }
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
                    AssertInFastLLM(srcCache.compressedBlocks == (startPos + segLen) / ratio,
                                    "DeepSeekV41: compressed cache is out of sync at layer " + std::to_string(layer));
                    if (srcCache.compressedBlocks > 0) {
                        compressedKV = &srcCache.compressedKV;
                        if (needIndexer) {
                            Data qIdxSeg, idxWeightsSeg, score;
                            Data *qIdx = sliceOf(qIdxAll, s, qIdxSeg);
                            V41RotaryQuant(*qIdx, rope, startPos, 1, false, 2, 32);
                            Data *idxWeightsScaled = sliceOf(idxWeightsAll, s, idxWeightsSeg);
                            V41IndexerScore(*qIdx, *idxWeightsScaled, srcCache.indexK, score);
                            if (dumpDebug) {
                                V41DumpTensor(*qIdx, "fl_layer" + std::to_string(layer) + "_idxq" + dumpSuffix);
                                V41DumpTensor(*idxWeightsScaled, "fl_layer" + std::to_string(layer) + "_idxw" + dumpSuffix);
                                V41DumpTensor(score, "fl_layer" + std::to_string(layer) + "_score" + dumpSuffix);
                            }
                            if (layer == candidate_source_layer_id && candidate_block_size > 0 && candidate_topk_blocks > 0) {
                                V41CandidateBlocks(score, candidate_block_size, candidate_topk_blocks, ratio, startPos,
                                                   segCandidate[s]);
                                segHasCandidates[s] = 1;
                                if (dumpDebug) {
                                    V41DumpTensor(segCandidate[s], "fl_layer" + std::to_string(layer) + "_cand" + dumpSuffix);
                                }
                            }
                            const bool useCandidates = segHasCandidates[s] && candidate_source_layer_id >= 0 &&
                                                       candidate_source_layer_id < layer;
                            V41IndexerTopK(score, useCandidates ? &segCandidate[s] : nullptr, indexTopK, ratio, startPos,
                                           std::max(1, candidate_block_size), segTopK[s]);
                        }
                        if (segTopK[s].dims.size() == 3) {
                            cmpIdx = &segTopK[s];
                        }
                    }
                }

                Data *attnOutSeg = single ? &attnOut : &segAttnOut[s];
                V41SparseAttention(*qSeg, *kvSeg, startPos > 0 ? &cache.windowKV : nullptr, compressedKV, cmpIdx,
                                   weight[pre + ".attn.attn_sink"], window_size, startPos, softmaxScale, *attnOutSeg);
                if (dumpDebug) {
                    std::string tag = "fl_layer" + std::to_string(layer);
                    V41DumpTensor(*qSeg, tag + "_q" + dumpSuffix);
                    V41DumpTensor(*kvSeg, tag + "_kv" + dumpSuffix);
                    V41DumpTensor(*attnOutSeg, tag + "_attn_o_raw" + dumpSuffix);
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
                Data kvSeg8;
                const Data *windowRows = kvSeg;
                if (fp8KV) {
                    V41QuantizeKV(*kvSeg, kvSeg8);
                    windowRows = &kvSeg8;
                }
                if (seg.spec != nullptr && seg.spec->deferWindow) {
                    // DSpark 校验：环形缓冲的写入推迟到接受长度确定之后。片段内的位置
                    // 一律从 chunkKV 读取，因此推迟写入不改变本次前向的任何结果，
                    // 也就不需要为回滚保存被覆盖的旧行。
                    seg.spec->windowKV[layer].CopyFrom(*windowRows);
                } else {
                    V41WindowStore(*windowRows, cache.windowKV, startPos, window_size);
                }
                V41RotaryQuant(*attnOutSeg, rope, startPos, 1, true, 0, 32);
                cache.totalLen += segLen;
            }

            Data *attnOutAll = single ? &attnOut : catSegments(segAttnOut, catTmp);
            DeepSeekV4WoA(*attnOutAll, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnProj);
            if (dumpDebug) {
                V41DumpTensor(attnInput, "fl_layer" + std::to_string(layer) + "_attn_in" + dumpSuffix);
                V41DumpTensor(*attnOutAll, "fl_layer" + std::to_string(layer) + "_attn_o" + dumpSuffix);
                V41DumpTensor(attnProj, "fl_layer" + std::to_string(layer) + "_attn" + dumpSuffix);
                if (segTopK[0].dims.size() == 3) {
                    V41DumpTensor(segTopK[0], "fl_layer" + std::to_string(layer) + "_topk" + dumpSuffix);
                }
            }
            DeepSeekV4HcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + "_hidden_attn" + dumpSuffix);
            }

            // ---- FFN (MoE)：整批 ----
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

        // ---- DSpark 校验片段：对本片段的每个位置都出贪心 token ----
        // 只在单片段（单请求）时启用，见 ForwardSingle。要求请求是简单贪心，因此
        // 这里的 RMSNorm + head + TopK 与 LLMSamplingBlock 的 allSimple 分支等价。
        if (numSegments == 1 && segments[0].spec != nullptr && segments[0].spec->wantAllGreedy) {
            Data allHidden, normed, allLogits, topk;
            V41HcApplyPre(*curHidden, preMix, allHidden);
            RMSNorm(allHidden, weight["norm.weight"], rms_norm_eps, normed);
            Linear(normed, weight["head.weight"], *GetEmptyData(), allLogits);
            ToDataType(allLogits, DataType::FLOAT32);
            TopK(allLogits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            const int stride = topk.dims[topk.dims.size() - 1];
            const float *topkData = (const float*)topk.cpuData;
            segments[0].spec->greedy.resize(seqlen);
            for (int i = 0; i < seqlen; i++) {
                segments[0].spec->greedy[i] = (int)(topkData[(uint64_t)i * stride] + 1e-3);
            }
            for (auto &seg : segments) {
                seg.state->totalLen += seg.seqlen;
            }
            return std::vector<int>{segments[0].spec->greedy[seqlen - 1]};
        }

        // ---- head（每个片段只取最后一个 token）----
        Data headInput;
        if (single && seqlen == 1) {
            V41HcApplyPre(*curHidden, preMix, headInput);
        } else {
            std::vector<Data> lastHidden(numSegments), lastPre(numSegments);
            for (int s = 0; s < numSegments; s++) {
                int end = segments[s].offset + segments[s].seqlen;
                Split(*curHidden, 1, end - 1, end, lastHidden[s]);
                Split(preMix, 1, end - 1, end, lastPre[s]);
            }
            Data hiddenTmp[2], preTmp[2];
            Data *hiddenAll = catSegments(lastHidden, hiddenTmp);
            Data *preAll = catSegments(lastPre, preTmp);
            V41HcApplyPre(*hiddenAll, *preAll, headInput);
        }

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(numSegments, 1);
        std::vector<GenerationConfig> generationConfigs = generationConfigsIn;
        for (auto &config : generationConfigs) {
            if (config.do_sample && config.top_k <= 1 && config.temperature > 1e-6f) {
                config.top_k = 5;
            }
        }
        LLMSamplingBlock(this, &headInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, numSegments, true, samplingSeqLens, samplingPastKeyValues,
                         generationConfigs, lastTokens, retLogits, ret);

        for (auto &seg : segments) {
            seg.state->totalLen += seg.seqlen;
        }
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
            v41StatesByFirstKey.erase((const void*)&pastKeyValues[0].first);
        }
        this->kvCacheId = 0;
        // 占位 KV 不反映真实占用；按模型几何估算每个 token 的长期缓存字节数
        // （压缩 KV + indexer key，均为 BF16；滑窗缓存长度固定，不计入），
        // 折算成 kvCacheDataType 的元素数供调度器估算上下文预算。
        long long bytesPerToken = 0;
        const long long kvRowBytes = this->kvCacheDataType == DataType::FP8_E4M3 ?
                                     (long long)head_dim_full + head_dim_full / 32 : (long long)head_dim_full * 2;
        for (int layer = 0; layer < block_cnt; layer++) {
            if (!isKvSource[layer]) {
                continue;
            }
            int ratio = std::max(1, compress_ratios[layer]);
            bytesPerToken += kvRowBytes / ratio;
            if (isIndexSource[layer]) {
                bytesPerToken += (long long)index_head_dim * 2 / ratio;
            }
        }
        long long unitBytes = std::max(1LL, (long long)GetDataBytes(this->kvCacheDataType, 1, 1));
        elementsInKVCachePerToken = std::max(1LL, (bytesPerToken + unitBytes - 1) / unitBytes);
        printf("finish.\n");
    }
}
