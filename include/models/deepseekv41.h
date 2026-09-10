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
//   7. mtp.* 为 DSpark 草稿模型（本实现暂不加载），vision.* / aligner.* 为视觉编码器（暂不加载）。
//
// 当前实现目标：在通用 CUDA/CPU 路径上跑通文本推理（单请求 prefill + decode），
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

    struct DeepSeekV41RequestState {
        std::vector<DeepSeekV41LayerCache> layers;
        std::vector<int> engramHistory;   // 每个已处理 token 的压缩 id（图像 token 为 -1）
        int totalLen = 0;
        int restoredLen = 0;              // 由前缀缓存恢复的 token 数（0 表示全新请求）
    };

    // 前缀缓存的一条记录：某段 token 序列处理完后的完整请求状态（张量放在 CPU）
    struct DeepSeekV41HistoryMemory {
        std::vector<int> tokens;          // 已经进入模型的 token（长度 == totalLen）
        int totalLen = 0;
        std::vector<DeepSeekV41LayerCache> layers;
        std::vector<int> engramHistory;
        long long flushTime = 0;
        int recordTimes = 0;
    };

    struct DeepSeekV41HistoryCacheManager {
        std::mutex locker;
        int maxRecordNum = 8;
        long long flushTime = 0;
        // Data 没有深拷贝赋值，记录一律通过 shared_ptr 持有，避免隐式拷贝造成别名
        std::map<std::vector<int>, std::shared_ptr<DeepSeekV41HistoryMemory> > memorys;

        void Record(const std::shared_ptr<DeepSeekV41HistoryMemory> &memory);
        // 按公共前缀长度从长到短列出候选（相同长度时记录更短的在前，更容易满足截断约束）；
        // 可截断性由模型侧检查
        std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> > GetCandidates(
                const std::vector<int> &inputTokens);
    };

    // 一次前向中的一个序列片段：属于哪个请求、从哪个位置开始、多少个 token、在拼接输入中的偏移
    struct DeepSeekV41Segment {
        std::shared_ptr<DeepSeekV41RequestState> state;
        int startPos = 0;
        int seqlen = 0;
        int offset = 0;
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
        // 状态同时按 &pastKeyValues（单请求 Forward）与 &pastKeyValues[0].first（调度器的多请求
        // ForwardBatch 只传每层 Data 指针）两把 key 索引；两者指向同一个 shared_ptr。
        std::mutex v41StateMutex;
        std::map<const void*, std::shared_ptr<DeepSeekV41RequestState> > v41States;
        std::map<const void*, std::shared_ptr<DeepSeekV41RequestState> > v41StatesByFirstKey;
        std::shared_ptr<DeepSeekV41RequestState> v41PendingRestoredState;   // TryRestoreHistoryCache 产生，
                                                                             // OnResponseContextCreated 接管
        DeepSeekV41HistoryCacheManager v41HistoryCache;

        std::shared_ptr<DeepSeekV41RequestState> GetOrCreateState(
                std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset);
        std::shared_ptr<DeepSeekV41RequestState> GetStateByFirstKey(const Data *firstKey);
        void RegisterState(const void *vectorKey, const void *firstKey,
                           const std::shared_ptr<DeepSeekV41RequestState> &state);

        // 前缀缓存：把请求状态快照到 CPU / 从快照恢复前 hitLen 个 token 的状态
        std::shared_ptr<DeepSeekV41HistoryMemory> SnapshotState(const DeepSeekV41RequestState &state,
                                                                const std::vector<int> &allTokens);
        std::shared_ptr<DeepSeekV41RequestState> RestoreState(const DeepSeekV41HistoryMemory &memory, int hitLen);
        // 检查快照能否截断到 len 个 token（滑窗环形缓存与压缩尾块的约束）
        bool CanTruncateHistory(const DeepSeekV41HistoryMemory &memory, int len) const;

        // 实际的前向：多个序列片段拼接成一个 token 流，Linear / MoE / Engram 查表按整批执行，
        // RoPE、压缩、indexer、稀疏注意力按片段分别执行。
        // inputEmbeds 非空时直接作为嵌入（[1, tokens, dim]，供视觉输入使用）；
        // imageMask 非空时标记每个 token 是否为图像 token（Engram 历史置 -1）。
        std::vector<int> ForwardSegments(
                std::vector<DeepSeekV41Segment> &segments,
                const Data &inputIds,
                const Data *inputEmbeds,
                const std::vector<int> *imageMask,
                const std::vector<GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector<std::vector<float>*> *retLogits,
                std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues);

        void LoadEngramMeta();
        void BuildEngramPrimes();

        // 计算 [tokens, (maxNgram-1)*heads] 的哈希行号
        void ComputeEngramHashes(int engramLayerIndex,
                                 const std::vector<int> &history, int startPos, int seqlen,
                                 std::vector<int64_t> &rows) const;

        // 从 FP8 表中取行，输出 BF16 [tokens, cols * headDim]
        void GatherEngramRows(int layer, const std::vector<int64_t> &rows, int tokens, Data &output);

        // 对一批片段做 Engram：各片段分别算哈希行号，查表 / wkv / 门控按整批执行
        void RunEngram(int layer, int engramLayerIndex, const std::vector<DeepSeekV41Segment> &segments,
                       Data &hiddenStates);
    };
}

#endif //FASTLLM_DEEPSEEKV41_H
