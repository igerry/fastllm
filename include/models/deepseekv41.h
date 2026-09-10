//
// DeepSeek-V4.1 系列模型（DeepSeek-V4.1-Flash）。
//
// 相对 DeepSeek-V4 的架构变化（参考 hfmodels/DeepSeek-V4.1-Flash/inference/model.py）：
//   1. Hyper-Connections 的 pre/post/comb 系数由"上一个子层"计算、"下一个子层"使用：
//      attention 使用上一层 FFN 产出的 pre，FFN 使用本层 attention 产出的 pre，
//      最后一层 FFN 产出的 pre 用于 lm_head 前的 hc_pre。第一层 attention 使用 one-hot pre。
//   2. 跨层共享压缩 KV：compress_ratios 取值 0 / 1 / 2。只有 kv_source_layer_ids 中的层
//      拥有 compressor 与 compress_kv_cache，其后（直到下一个 source 之前）的层直接读取
//      该 cache；index_source_layer_ids 中的层运行 indexer，其它层复用最近 index source
//      的 top-k 结果。ratio 1 的 compressor 是纯投影（无 gate），ratio 2 做 softmax 池化。
//   3. 两级 indexer：candidate_source_layer_id 层先按 candidate_block_size 分块选出
//      candidate_topk_blocks 个块，之后的 index source 只在这些块内部做 top-k。
//      indexer 的 key 由 compressor 的 latent 经 wk + k_norm 派生，不再有独立 compressor。
//   4. Engram：在 engram_layer_ids 层之前，对 residual stream 做 n-gram 哈希查表
//      （embed 为 FP8 + 逐行 32 列一组的 UE8M0 scale），经 wkv 得到 key/value 并做门控写回。
//      哈希基于 tokenizer 归一化后的压缩 token id（engram_token_map，由 Python 侧生成）。
//   5. 无 hash 路由层；gate 使用 bias（noaux_tc），图像 token 使用 bias_vl。
//   6. 稠密权重 FP8 块大小 32x32，专家 FP4（沿 K 每 32 个一组 UE8M0 scale）。
//   7. mtp.* 为 DSpark 草稿模型（本实现暂不加载）；vision.* / aligner.* / image_* 为视觉编码器
//      （ViT + 3x3 下采样 aligner，实现见 src/models/deepseekv41_vision.cpp），图像 token 的嵌入
//      由 ForwardMultimodal 写入 input_ids 中 image_token_id 的位置。
//
// 当前实现目标：在通用 CUDA/CPU 路径上跑通文本 / 图文推理（单请求 prefill + decode），
// 专家与 Engram 表放在 CPU 内存，注意力层放在 GPU。
//

#ifndef FASTLLM_DEEPSEEKV41_H
#define FASTLLM_DEEPSEEKV41_H

#include "deepseekv4.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace fastllm {
    // 单层的推理缓存
    struct DeepSeekV41LayerCache {
        int totalLen = 0;             // 已经进入本层的 token 数
        Data windowKV;                // [windowSize, headDim] BF16 环形滑窗 KV，row = pos % windowSize

        // 以下仅 kv source 层使用
        Data compressedKV;            // [capacity, headDim] BF16，已加 RoPE 与 FP4 伪量化
        int compressedBlocks = 0;     // 已经写入的压缩块数
        Data indexK;                  // [capacity, indexHeadDim] BF16，indexer 的 key
        Data rawTailKV;               // FP32 [rawTail, headDim]，尚未凑满一组的 compressor 原始输入
        Data rawTailScore;            // FP32 [rawTail, headDim]
        int rawTail = 0;
    };

    // 一张图在 prompt 中占据的 span（含 image_start / image_newline / image_end）及其嵌入
    struct DeepSeekV41ImageSpan {
        int start = 0;
        int length = 0;
        Data embeds;                      // CPU FLOAT32 [length, dim]
    };

    struct DeepSeekV41RequestState {
        std::vector<DeepSeekV41LayerCache> layers;
        std::vector<int> engramHistory;   // 每个已处理 token 的压缩 id（图像 token 为 -1）
        int totalLen = 0;
        // 图文请求：由 OnResponseContextCreated / ForwardMultimodal 记录，首个 prefill 块编码为 imageSpans
        const std::map <std::string, std::vector <Data*> > *pendingMultimodal = nullptr;
        bool imagesEncoded = false;
        std::vector<DeepSeekV41ImageSpan> imageSpans;
    };

    // Engram 哈希元数据（由 tokenizer 归一化派生，Python 侧生成 JSON，此处加载）
    struct DeepSeekV41EngramMeta {
        bool loaded = false;
        std::vector<int> tokenMap;                       // 原始 token id -> 压缩 id
        int compressedVocabSize = 0;
        int padCompressedId = 0;                         // engram_pad_token_id 映射后的压缩 id
        std::vector<std::vector<int64_t> > multipliers;  // [engramLayer][maxNgram]
        std::vector<std::vector<int64_t> > primes;       // [engramLayer][(maxNgram-1)*heads]
        std::vector<std::vector<int64_t> > offsets;      // [engramLayer][(maxNgram-1)*heads]
    };

    class DeepSeekV41Model : public DeepSeekV4Model {
    public:
        DeepSeekV41Model();

        ~DeepSeekV41Model() override;

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType> > >
                GetTensorMap(const std::vector<std::string> &tensorNames) override;

        void OnModelWeightsLoaded() override;

        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr) override;

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        // 图文前向：multimodalInput 由 Python 侧的 deepseek_v41_multimodal.py 构造：
        //   "pixel_values": 每张图一个 FLOAT32 [nPatches, 3 * patch * patch]
        //   "image_grid":   INT32 [numImages, 3]，每行 (span 起始位置, nVitH, nVitW)
        // 图像 span 的嵌入在请求的第一个 prefill 块编码并缓存在请求状态里，之后每个块（无论由调度器
        // 还是本函数切分）按位置重叠写入，因此 span 可以跨块；decode 步骤退化为普通 Forward。
        std::vector <int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr) override;

        void WarmUp() override;

        bool TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) override;
        void TryRecordHistoryCache(const std::vector<int> &allTokens) override;
        void TryRecordResponseContext(ResponseContext *context) override;
        void OnResponseContextCreated(ResponseContext *context) override;
        void OnResponseContextRemoved(ResponseContext *context) override;
        bool UseGenericHistoryCache() const override { return false; }
        bool UseModelSpecificScheduler() const override { return false; }

    protected:
        // -------- 跨层共享 --------
        std::vector<int> kv_source_layer_ids;
        std::vector<int> index_source_layer_ids;
        int candidate_source_layer_id = -1;
        int candidate_topk_blocks = 0;
        int candidate_block_size = 0;
        float gate_temp = 1.0f;
        int image_token_id = -1;

        // -------- 视觉编码器（deepseekv41_vision.cpp）--------
        int vision_n_layers = 0;          // 0 表示没有视觉塔
        int vision_dim = 1024;
        int vision_n_heads = 16;
        int vision_inter_dim = 2816;
        int vision_patch_size = 14;
        int vision_downsample_ratio = 3;
        float vision_rope_theta = 10000.0f;
        float vision_norm_eps = 1e-6f;
        bool VisionEnabled() const { return vision_n_layers > 0; }
        void InitVisionParams();
        bool IsVisionTensor(const std::string &name) const;
        // ViT + aligner：patches FLOAT32 [nVitH * nVitW, 3 * patch * patch]，
        // 输出 CPU FLOAT32 [ceil(nVitH / r) * ceil(nVitW / r), dim]
        void EncodeImage(const Data &patches, int nVitH, int nVitW, Data &output, const std::string &dumpPrefix = "");
        // 编码 multimodalInput 中的全部图像，得到每个 span 的嵌入（分隔符 + aligner 输出）并存入 state
        void EncodeImageSpans(const std::map <std::string, std::vector <Data*> > &multimodalInput,
                              DeepSeekV41RequestState &state);
        // 若本块 [startPos, startPos + seqlen) 与某个图像 span 重叠：embeds = 文本嵌入并写入图像嵌入
        //（CPU FLOAT32 [1, seqlen, dim]），imageMask[i] = 1 表示图像 token；返回是否有重叠
        bool PrepareImageEmbeds(const Data &inputIds, int startPos, DeepSeekV41RequestState &state,
                                Data &embeds, std::vector<int> &imageMask);

        // 每层派生信息
        std::vector<int> kvSourceOf;      // 本层读取哪一层的压缩 KV（-1 表示纯滑窗）
        std::vector<int> indexSourceOf;   // 本层复用哪一层的 top-k
        std::vector<char> isKvSource;
        std::vector<char> isIndexSource;

        // -------- Engram --------
        std::vector<int> engram_layer_ids;
        std::vector<int64_t> engram_num_embeddings;
        int engram_max_ngram_size = 4;
        int engram_vocab_size = 0;
        int engram_n_heads = 0;
        int engram_head_dim = 0;
        int engram_pad_token_id = 2;
        int engram_compressed_vocab_size = 0;
        DeepSeekV41EngramMeta engramMeta;
        std::vector<std::shared_ptr<void> > engramTables;   // 每个 engram 层一张表（实现见 cpp）

        // -------- 请求状态 --------
        std::mutex v41StateMutex;
        std::map<const void*, std::shared_ptr<DeepSeekV41RequestState> > v41States;

        std::shared_ptr<DeepSeekV41RequestState> GetOrCreateState(
                std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset);

        // 单序列前向的实际实现。inputEmbeds 非空时以其（CPU FLOAT32 [1, seqlen, dim]）代替 embedding 查表；
        // imageMask 非空时（长度 seqlen，1 = 图像 token）路由改用 gate.bias_vl。
        std::vector <int> ForwardSingle(
                const Data &inputIds,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *logits,
                const Data *inputEmbeds,
                const std::vector<int> *imageMask);

        void LoadEngramMeta();
        void BuildEngramPrimes();

        // 计算 [tokens, (maxNgram-1)*heads] 的哈希行号
        void ComputeEngramHashes(int engramLayerIndex,
                                 const std::vector<int> &history, int startPos, int seqlen,
                                 std::vector<int64_t> &rows) const;

        // 从 FP8 表中取行，输出 BF16 [tokens, cols * headDim]
        void GatherEngramRows(int layer, const std::vector<int64_t> &rows, int tokens, Data &output);

        void RunEngram(int layer, int engramLayerIndex, const std::vector<int> &history,
                       int startPos, int seqlen, Data &hiddenStates);
    };
}

#endif //FASTLLM_DEEPSEEKV41_H
