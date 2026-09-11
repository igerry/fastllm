# DeepSeek-V4.1-Flash 支持说明

[返回 README](../README.md) · [DeepSeek-V4 部署指南](deepseek.md) · [混合推理](mixforward.md)

本文记录 FastLLM 对 DeepSeek-V4.1 系列（`model_type = deepseek_v41`，目前为 DeepSeek-V4.1-Flash）的支持范围、
架构差异、启动方式与验证方法。实现位于：

- `include/models/deepseekv41.h` / `src/models/deepseekv41.cpp`：模型（继承 `DeepSeekV4Model`，复用其 MoE / HC-post / WoA / 采样等基础设施）
- `src/models/deepseekv41_vision.cpp`：视觉编码器（ViT + aligner）与图文前向 `ForwardMultimodal`
- `src/devices/cpu/deepseekv41ops.cpp`：V4.1 专用算子的 CPU 参考实现
- `src/devices/cuda/models/deepseekv41-kernels.cu`：对应的 CUDA kernel（面向 SM86 等无 FP8 tensor core 的设备；
  稀疏注意力与 indexer 打分在 SM80+ 上走 BF16 mma，其余算子为 FP32）
- `src/models/deepseekv41_dspark.cpp`：DSpark 投机解码（草稿层 `mtp.*`、校验与回滚）
- `tools/fastllm_pytools/deepseek_v41_engram.py`：Engram 哈希元数据生成
- `tools/fastllm_pytools/encoding_dsv41.py`：官方 V4.1 prompt 编码（vendored）
- `tools/fastllm_pytools/deepseek_v41_multimodal.py`：图像动态分辨率预处理（移植自官方 `image_processor.py`）与图像占位符展开
- `test/basic/deepseek_v41_reference.py`：与官方 `inference/model.py` 的端到端数值对齐测试
- `test/basic/deepseek_v41_vision_reference.py`：图文输入的端到端对齐测试（含真实 ViT 权重验证）
- `test/basic/deepseek_v41_dspark.py`：DSpark 的精确性（开 / 关输出一致）与回滚测试

## 相对 DeepSeek-V4 的架构变化

| 项目 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| Hyper-Connections | 每个子层自己算 pre/post/comb 并立即使用 | pre 由上一个子层计算、下一个子层使用；最后一层 FFN 的 pre 用于 lm_head 前的折叠 |
| 压缩注意力 | 每层独立 compressor（ratio 4 / 128，带 ape、overlap） | `compress_ratios` 取 0 / 1 / 2；只有 `kv_source_layer_ids` 中的层拥有 compressor 与压缩 KV cache，其后各层直接读取（跨层共享） |
| Indexer | 每个 CSA 层有独立的 128 维 compressor + Hadamard 旋转 | key 由 compressor latent 经 `wk + k_norm` 派生；只有 `index_source_layer_ids` 层运行 indexer，其它层复用最近的 top-k |
| 两级 top-k | 无 | `candidate_source_layer_id` 层先按 `candidate_block_size` 分块选出 `candidate_topk_blocks` 个块，之后的 index source 只在候选块内选 top-k |
| Engram | 无 | `engram_layer_ids` 层前对 residual stream 做 n-gram 哈希查表（每层约 100 GB 的 FP8 表）并门控写回 |
| 路由 | 前若干层 hash 路由 | 全部 noaux_tc，`gate.bias`；图像 token 使用 `gate.bias_vl` |
| q 处理 | 额外 RMS 归一 | 无 |
| 权重格式 | FP8 128x128 块 scale | 稠密与共享专家 FP8 32x32 块 UE8M0 scale；路由专家 FP4（沿 K 每 32 个一组 UE8M0 scale） |
| 附加模块 | MTP / DSpark | DSpark 草稿层（mtp.*，markov head 为 embed + head、草稿层 128 专家 top-3、目标层取自主干）、视觉编码器（vision.* / aligner.* / image_start / image_end / image_newline） |

## 当前支持范围

已实现并通过数值对齐测试：

- 文本推理（prefill + decode，含分块 prefill），CUDA 与 CPU 两套算子路径；
- 多请求批量 decode：多个请求的 token 拼成一个序列共享一次前向（Linear / MoE / Engram 查表按整批执行，
  注意力与压缩 KV 按请求分别执行），见下文"多请求与前缀缓存"；
- 前缀缓存 / 多轮对话复用（`--cache_history true`）；
- 跨层共享压缩 KV（ratio 1 / 2）、两级 indexer top-k、Engram、Hyper-Connections、sqrt-softplus 路由；
- 与官方实现一致的 FP8 / FP4 伪量化（窗口 KV、压缩 KV、indexer q/k）；
- 真实 checkpoint 的权重格式（FP8 32x32、FP4 路由专家、FP8 + UE8M0 的 Engram 表）；
- 图像输入（OpenAI 接口的 `image_url`）：ViT + aligner、图像 token 的 `gate.bias_vl` 路由与 Engram 掩码，详见下文；
- DSpark 投机解码（`mtp.*` 草稿层），详见下文"DSpark 投机解码"。

尚未实现：

- CUDA Graph、张量并行等 V4 已有的性能特性；
- DSpark 与批量 decode 的组合（批内不产生候选，只保持草稿缓存同步）、DSpark 与采样 / 图文请求的组合。

## Engram 元数据

Engram 的哈希基于 tokenizer 归一化（NFKC / NFD / 去重音 / 小写 / 空白折叠）后的"压缩 token id"。
该映射依赖 HuggingFace `tokenizers` 的 normalizer，因此由 Python 侧生成、C++ 侧加载：

```bash
python -m ftllm.deepseek_v41_engram /path/to/DeepSeek-V4.1-Flash
# 生成 /path/to/DeepSeek-V4.1-Flash/engram_meta.json（压缩词表大小应为 99092）
```

通过 `ftllm` 启动时会自动生成（模型目录只读时写到 `~/.cache/fastllm/engram/`），也可以用环境变量
`FASTLLM_DSV41_ENGRAM_META=/path/engram_meta.json` 显式指定。素数桶布局由 C++ 侧按官方算法推导，
并与 `engram_num_embeddings` 做一致性校验。

Engram 表（两层，各约 100 GB）不经过通用加载器，而是由模型直接从 safetensors 读入内存，
以 FP8 + UE8M0 scale 原样保存；查表在 CPU 完成，`wkv` 投影与门控在 GPU 完成。
设置 `FASTLLM_DSV41_ENGRAM_MMAP=1` 可改为 mmap（首次访问慢，节省常驻内存）。

## 启动

以 2 x 24 GB GPU + 大内存主机为例（专家与 Engram 表放在 CPU 内存）：

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa \
  --dtype float16 \
  --chunked_prefill_size 4096
```

- 没有 FP8 tensor core 的 GPU（如 SM86）请使用 `--dtype float16`，稠密 FP8 权重会在加载时按 32x32 块 scale 反量化；
- `--kv_cache_dtype` 控制长期 KV 缓存（压缩 KV + indexer key）的存储精度，见下面的“KV 缓存存储精度”一节。
  默认 BF16，可选 `fp8_e4m3`（1650 B/token）与 `fp4_e2m1`（890 B/token，与官方一致，且数值无损）；
- `ftllm` launcher 的自动配置按 config 计算 V4.1 的常驻内存下界（FP4 专家约 289 GB + Engram 表约 203 GB +
  稠密部分），权重文件不全时也不会低估；主机内存不足以放下专家与 Engram 表时会退到 `moe_device=disk`；
- 单路 CPU 机器可用 `--moe_device cpu`；
- 内存需求：Engram 表约 200 GB + 路由专家（FP4）约 270 GB + 加载临时空间；
- 首次启动会生成 `engram_meta.json`（约 1 分钟）并读入两张 Engram 表。

### 实测（DeepSeek-V4.1-Flash 真实权重，2026-09-11）

主机：EPYC 7C13（128 线程）+ 943 GB 内存 + 2 x RTX 3090 Ti（SM86，24 GB），权重在共享盘上（478 GB，48 个分片）。
启动参数即上面的推荐配置（`--device cuda --moe_device numa --dtype float16 --chunked_prefill_size 4096 -t 64`）。

| 指标 | 实测 |
| --- | --- |
| 加载耗时 | 约 7 分钟（含两张 Engram 表各 101.4 GB 读入内存） |
| 显存 | 15.9 GB，单卡即可（稠密 FP8 反量化为 float16 后） |
| 内存 | 约 521 GB 常驻 |
| 图像 | 448x336 的图 210 个 prompt token，端到端 6 秒 |

prefill 与 decode（同一台机器、同一组提示词，BF16 mma kernel 上线前后对照）：

| 上下文 | prefill 旧 | prefill 新 | decode 旧 | decode 新 |
| --- | --- | --- | --- | --- |
| 14 | 1.9 s | 2.2 s | 194 ms/token | 164 ms/token |
| 974 | 5.7 s | 4.8 s | 302 ms/token | 153 ms/token |
| 8014 | 35.9 s | 16.9 s | 423 ms/token | 234 ms/token |
| 32014 | 207.3 s | 66.4 s | 319 ms/token | 132 ms/token |

单步 decode 的算子分解（`FASTLLM_PRINT_PROFILE=1 FASTLLM_CUDA_SYNC=1`，短上下文）显示 `MergeMOE`
（CPU 上的 FP4 路由专家）占绝对多数：新 kernel 上线前为 137 ms / 190 ms。因此短上下文的 decode
主要由 CPU 专家决定，长上下文与 prefill 才由注意力与 indexer 决定。

CPU 专家的当前瓶颈：融合的 FP4 GEMM 只有 AVX512-BF16 实现，在没有 AVX512 的机器（如 Zen 3 的 EPYC 7C13）
上回退为"先把 FP4 解量化到临时 BF16 缓冲，再做普通 BF16 GEMM"，读写放大数倍。见文末的未完成项。

行为验证（贪心解码）：中英文常识、算术、代码生成、逻辑推理均正确；34k token 上下文的"大海捞针"命中
（该长度会激活候选块两级 top-k）；工具调用能正确产出 `tool_calls`；图像输入能正确描述图中的形状与颜色。

## KV 缓存存储精度（`--kv_cache_dtype`）

长期 KV 缓存只有两部分：**压缩 KV**（`kv_source_layer_ids` 各层，每 `compress_ratio` 个 token 一行，
`head_dim` = 512）与 **indexer key**（同样的层，`index_head_dim` = 128）。滑窗缓存长度固定为
`sliding_window`，不随上下文增长，不计入每 token 开销。

写进缓存之前，这两者都已经过与官方一致的伪量化（`DeepSeekV41RotaryQuant` 的 `quantMode`）：

| 张量 | 伪量化网格 | 分组 | scale 编码 |
| --- | --- | --- | --- |
| 压缩 KV | FP4 E2M1 | 每 16 个通道 | E4M3 |
| indexer key | FP4 E2M1 | 每 32 个通道 | UE8M0（2 的幂） |
| 滑窗 KV | FP8 E4M3 | 每 32 个通道 | UE8M0 |

也就是说值本来就落在 FP4 / FP8 网格上，**按 FP4 存储是无损的**。三档存储的行布局与开销：

| `--kv_cache_dtype` | 压缩 KV 行 | indexer key 行 | 每 token | 相对 BF16 | 精度 |
| --- | --- | --- | --- | --- | --- |
| 默认（BF16） | 1024 B | 256 B | **3200 B** | 1.00x | 基准 |
| `fp8_e4m3` | 528 B（512 E4M3 + 16 UE8M0） | 132 B | **1650 B** | 0.52x | 压缩 KV 有不超过 2^-4 的相对舍入 |
| `fp4_e2m1` | 288 B（256 打包 E2M1 + 32 E4M3） | 68 B（64 打包 E2M1 + 4 UE8M0） | **890 B** | 0.28x | **逐 bit 无损** |

890 B/token 与官方实现一致。每 token 2.5 行压缩 KV（层 2/8/14 的 `compress_ratio` 是 2，层 20 是 1）：
`2.5 x 288 + 2.5 x 68 = 890`。

- **FP4 无损的原因**：`DeepSeekV41QuantizeKV` 的块 scale 推导（`DeepSeekV41BlockScale`）与伪量化
  (`DeepSeekV41FakeQuantRow`) 是同一份代码，分组大小与 scale 编码也完全对齐，所以对已经伪量化过的行
  是幂等的；而 `q * scale`（q 至多 3 个有效位、scale 是 E4M3 或 2 的幂）在 BF16 上也是精确的。
  实测：开启伪量化时 FP4 存储与 BF16 存储的 logits **逐 bit 相同**（`test/basic/deepseek_v41_kv_cache_dtype.py`）。
- **FP8 为什么有损**：压缩 KV 的网格是「FP4 值 x E4M3 scale」，而 FP8 存储用的是 2 的幂块 scale，
  两者的网格对不上，会引入一次真实的舍入。它的用途是在不支持 FP4 打包读取的场合省一半显存。
- 滑窗 KV 在 `fp8_e4m3` 与 `fp4_e2m1` 两档下**都按 FP8 存储**（它本来就在 FP8 网格上，改 FP4 会真的损失精度），
  三种行布局可以在同一次前向里混用，读路径按行宽自动识别。
- 缓存行统一放在 `INT8` 的 `Data` 里，形状 `[b, rows, rowBytes]`；FP4 打包为一个字节两个 code，
  低 4 位是偶数下标。CUDA 侧在把候选装进共享内存时就地解成 BF16 片段，再进 `mma.sync`，
  `FASTLLM_DSV41_LEGACY_ATTN` / `FASTLLM_DSV41_LEGACY_INDEXER` 的标量回退路径同样支持。

### 怎么选

- **长上下文 / 高并发**：用 `fp4_e2m1`。同样显存能装的上下文约为 BF16 的 3.6 倍，且数值与 BF16 完全一致，
  没有任何精度代价；长上下文下稀疏注意力每步读取的字节数同比下降，decode 也有正收益。
- **默认（BF16）**：短上下文、显存不紧张时省掉打包/解包的一点开销。
- `fp8_e4m3`：介于两者之间，但既比 FP4 大又比 FP4 差，现在没有明显适用场景，保留是为了兼容既有部署。

### 实测（迷你模型，KV 几何与真实 config 相同：`kv_source_layer_ids` [2, 8, 14, 20]、ratio 2/2/2/1，单卡 3090 Ti）

`test/basic/deepseek_v41_kv_cache_dtype.py` 用同一段提示词跑三档，实测的长期 KV 占用与逐 bit 比较：

| | 每 token | 32k 上下文的实际缓存 | 1 GiB 能装的上下文 | 与 BF16 的 logits |
| --- | --- | --- | --- | --- |
| BF16 | 3200 B | 102.6 MB | 33.5 万 token | 基准 |
| `fp8_e4m3` | 1650 B | 52.9 MB | 65.1 万 token | 有差（压缩 KV 有舍入） |
| `fp4_e2m1` | 890 B | 28.5 MB | **120.6 万 token** | **逐 bit 相同** |

速度（同一台机器，prefill / decode）：

| 上下文 | BF16 | `fp4_e2m1` |
| --- | --- | --- |
| 1000 | 0.32 s / 9.35 ms per token | 0.32 s / 9.64 ms |
| 8000 | 0.56 s / 10.12 ms | 0.64 s / 9.41 ms |
| 32000 | 1.53 s / 12.32 ms | 2.03 s / 10.09 ms |

上下文越长，decode 越占便宜（32k 时快 18%），因为稀疏注意力与 indexer 每步读取的字节数减半以上。
prefill 反过来会慢一些（32k 时慢约 30%）：迷你模型里 MoE 很小，FP4 的逐元素解包（相对 BF16 的
`float4` 整块搬运）成了可见开销；真实模型的 prefill 由 MoE 主导，这部分占比会小得多。
把解包改成按 4 字节一组的向量化 nibble 展开可以再优化，尚未做。

`FASTLLM_DSV41_KV_STATS=1` 会在每次前向后打印实际占用的长期 KV 字节数与每 token 均值，便于核对。

## 多请求与前缀缓存

### 批量 decode

`DeepSeekV41Model::ForwardSegments` 把若干"片段"（请求状态 + 起始位置 + token 数）拼成一个 token 流：
embedding、Hyper-Connections、各 Linear、路由、MoE 与 Engram 查表按整批执行，RoPE、压缩 KV 追加、
indexer top-k、稀疏注意力与滑窗写入按片段分别执行。通用调度器的多请求 `ForwardBatch` 直接走这条路径，
单请求 `Forward` 是它的单片段特例；含长 prefill 片段的混合批次退回为逐个前向。

每个请求的缓存（`DeepSeekV41RequestState`）与 `ResponseContext` 绑定，请求结束或 abort 时释放。
在迷你模型上，8 并发 decode 与逐个请求相比，token 序列的差异仅来自 GEMM 按 batch 选核带来的
BF16 舍入（同一请求换不同的批次伙伴 / 批内位置，logits 逐 bit 一致）；单卡 3090Ti 上迷你模型的
decode 吞吐从 209 tok/s（1 并发）提高到 573 tok/s（8 并发）。

### 前缀缓存

启动时加 `--cache_history true`。请求结束时把每层 `windowKV`、`compressedKV`、`indexK`、`rawTail`
与 Engram 历史快照到 CPU 内存（LRU，默认保留 8 条），新请求按最长公共前缀查找并恢复，只对新增 token
做 prefill。多轮对话中只要客户端原样回传上一轮的回复，通常就是精确命中。

恢复长度受模型结构约束：滑窗缓存是只保留最后 `window_size` 个位置的环形缓冲，因此只能恢复到记录
长度 T 或 T-1（记录不超过 `window_size` 时可以任意截断）；ratio-2 压缩层凑不满一组的原始尾块只在
精确命中时可用，其它情况要求恢复长度为偶数。不满足约束的候选会被跳过（退回完整 prefill）。

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` | 关闭前缀缓存 |
| `FASTLLM_DSV41_PREFIX_CACHE_DEBUG` | 打印命中 / 记录日志 |
| `FASTLLM_DSV41_PREFIX_CACHE_MIN_TOKENS` | 最短命中长度（默认 16） |
| `FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS` | 最多保留的记录数（默认 8，也可用 `FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS`） |

迷你模型上精确命中与 T-1 截断命中后的贪心输出与不中断的原请求逐 bit 一致；800 token 提示词的
首 token 延迟从 39–54 ms 降到 6 ms。

## DSpark 投机解码

`config.json` 的 `text_config` 里 `dspark_block_size > 0` 且 checkpoint 带 `mtp.*` 权重时可以开启。
启动加 `--speculative_algorithm dspark --dspark 5`（5 = `dspark_block_size`，也可以更小，
每轮少校验几个候选）：

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa --dtype float16 \
  --speculative_algorithm dspark --dspark 5
```

`ftllm` 的自动配置（launcher）在 `enable_speculative_decoding` 时会识别 V4.1 的内置 DSpark，
按 checkpoint 的训练 block size 填 `--draft_tokens`。不加 `--dspark` / `--draft_tokens` 时
不加载 `mtp.*`（省下约 30 GB 权重）。

### 结构

`mtp.0/1/2` 是三个与主干同构的草稿层：`compress_ratios` 在这三层是 0（纯滑窗注意力），
MoE 是 `dspark_n_routed_experts` = 128 专家 top-3，embedding 与 lm_head 与主干共享。
另外 `mtp.0` 有 `main_proj` / `main_norm`，`mtp.2` 有 `norm`、`markov_head`（`embed` + `head`，
秩 256）与 `confidence_head`（输入 `dim + 256`）。

草稿层的滑窗 KV 不来自草稿 token 自身，而来自目标模型：`dspark_target_layer_ids`（[37, 38, 39]）
各层 attention 的输入（对 4 份 Hyper-Connections 取均值）拼成 `3 * dim` 后经 `main_proj` / `main_norm`
得到 `main_x`，每个草稿层再用自己的 `wkv` / `kv_norm` 把它写进一行滑窗 KV。也就是说每个已提交
位置在草稿侧只有一行 KV，草稿模型不需要重跑主干。

一次 proposal：把 `block_size` 个位置的输入 token 置为 `dspark_noise_token_id`（第 0 个位置放锚点
token，即目标模型刚产出、还没进 KV 缓存的那个 token），一次前向产出 `block_size` 组 logits；
再用 markov head 逐位置做 bigram 修正后贪心采样，得到 `block_size` 个候选 token；
`confidence_head` 对每个位置给出一个 sigmoid 后的置信度。

### 校验与回滚

候选与锚点拼成一个 `1 + N` 长度的片段一次喂给目标模型（`ForwardSegments` 天然支持一次多 token），
逐位置贪心比对，第一个不匹配处截断。接受 n 个候选时这一轮提交 n + 1 个 token、产出 n + 1 个输出
token：第一个立刻返回，其余进入请求的待发队列，调度器之后每轮直接出队，不再前向。

校验前向按完整 block 更新缓存，接受长度确定后要把多算的部分退回：

- 滑窗环形缓冲：写入**延后**到接受长度确定之后。片段内的位置在稀疏注意力里一律从 `chunkKV` 读取，
  推迟写入不改变本次前向的任何结果，因此也不需要为回滚保存被覆盖的旧行；
- 压缩 KV 与 indexer key：按接受后的长度重新截断行数，并用保存下来的 compressor 原始输入流
  （旧 `rawTail` + 本次新行）重建 `rawTail`；
- Engram 历史与各层 `totalLen`：截断到接受后的长度。

投机解码是精确的：接受的 token 就是目标模型在同一次前向里算出的贪心 token，因此开启 DSpark 与
关闭时的贪心输出一致。唯一的差异来源与批量 decode 相同——一次多 token 的前向与逐 token 前向会
选到不同的 GEMM kernel，BF16 舍入可能让几乎并列的 argmax 翻转（这一点不开 DSpark 时，
一次大 prefill 与逐 token 解码之间同样存在）。

### 限制

- 只对**简单贪心**请求生效：`do_sample` / `top_k > 1` / 重复惩罚 / 工具约束 / `output_logits` /
  `output_token_least` 中任何一项打开，该请求就退回普通解码（草稿侧仍然保持滑窗同步）；
- 只在**单请求**前向里产生候选。批量 decode 的那一轮不投机，但仍然采集 main hidden，
  让草稿滑窗跟上目标缓存；已经校验通过的 token 在批量路径里也能正常出队；
- 图文请求不投机；
- 前缀缓存命中恢复出来的前缀没有草稿侧的滑窗（`main_x` 无法从目标缓存反推），
  草稿注意力只看得到恢复之后新增的位置，接受率会在最初的 `window_size` 个 token 内偏低；
- 请求在待发队列还没取完时结束（EOS / 长度上限），这一轮多算的 token 会让缓存长度超过
  `allTokens`，该请求的前缀缓存记录会被跳过；
- 尚未接入 CUDA Graph 与张量并行。

### 实测

单卡 3090 Ti、迷你模型（6 层 + 3 个草稿层、2 专家、随机权重）：

| 场景 | 结果 |
| --- | --- |
| 开 / 关 DSpark 的贪心输出 | 一致（30 个 token 中 3 处 BF16 并列翻转，重新锚定后完全一致） |
| 注入候选构造接受 0 / 1 / 2 / 3 / 5 个 | 接受长度与构造完全一致，回滚后的续写与基准一致 |
| 随机权重下的接受率 | 0%（草稿模型是随机初始化的，只验证正确性，不代表真实接受率） |
| 吞吐 | 190 → 100 tok/s（草稿层占迷你模型的一半，且接受率为 0，属最坏情况） |

真实 checkpoint 的 `mtp.*`（3 个 stage × 128 专家，共 2401 个张量）已核对：加载器需要的 1221 个
张量全部存在，量化格式与主干一致（稠密 FP8 32x32 + UE8M0 scale、路由专家 FP4 沿 K 每 32 个一组），
`markov_head.embed/head` 为 BF16 `[129280, 256]`、`confidence_head.proj` 为 BF16 `[1, 5376]`。
真实权重下的接受率与吞吐尚未测（需要双卡 + 全量权重）。

### 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSPARK_TOKENS` | 每轮校验的候选数（由 `--dspark` / `--draft_tokens` 设置，不要直接设） |
| `FASTLLM_DSPARK_CONFIDENCE_THRESHOLD` | 置信度低于该值的候选之后不再校验；0 表示总是用满 block |
| `FASTLLM_DSPARK_STATS_FILE` | 每轮校验追加一行 `轮次 候选数 接受数`（测试用） |
| `FASTLLM_DSPARK_FORCE_DRAFTS` | 用文件里的候选替换模型的候选，构造指定的接受长度（测试用） |
| `FASTLLM_DSV41_DEBUG_STATE` | 每次前向后导出各层缓存长度（对比回滚是否正确） |

## 图像输入

`config.json` 顶层的 `vision_n_layers > 0` 时加载视觉编码器：32 层 ViT（1024 维、16 头、patch 14、2D RoPE、SwiGLU MLP）
+ aligner（3x3 下采样后 `w1 -> GELU -> w2` 投影到 5120 维）+ 三个学习到的分隔符嵌入。权重保持 float16（不参与低比特量化），
以 float32 激活在执行器所在设备上计算，长序列注意力按 1024 个 query 分块。

处理流程（与官方 `image_processor.py` / `model.py` 一致）：

1. Python 侧（`deepseek_v41_multimodal.py`）：`encode_messages(..., return_multi_modal_data=True)` 渲染带
   `<｜deepseek_image｜>` 占位符的 prompt；每张图按动态分辨率规则缩放 / 灰色补边（`vision_min_pixels` 295936、
   `vision_max_n_token` 1024），切成 `n_vit_h x n_vit_w` 个 patch，占位符展开为
   `[image_start] + ([IMAGE] * n_llm_w + [image_newline]) * n_llm_h + [image_end]`（每个位置都是 `image_token_id`），
   patch 与每张图的 `(起始位置, n_vit_h, n_vit_w)` 作为 payload 传给 C++；
2. C++ 侧：请求创建时把多模态输入记到请求状态（`OnResponseContextCreated` / `ForwardMultimodal`），
   第一个 prefill 块对每张图做 ViT + aligner，把结果与分隔符嵌入组成 span 嵌入缓存起来；之后每个 prefill 块
   （无论由调度器还是模型自己按 `chunked_prefill_size` 切分，span 可以跨块）把与 span 重叠的位置换成图像嵌入，
   图像 span 内的 token 路由使用 `gate.bias_vl`、Engram 对其不做查表也不参与 n-gram；decode 步骤与纯文本相同。

用法：与其它多模态模型相同，OpenAI 接口的 `image_url` 支持 http(s)、`data:image/...;base64` 与 `file://`：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "v41", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}]}]}'
```

限制：

- 只支持图像，不支持视频；一张图最多 1024 个 token（约 9216 个 ViT patch），多张图按出现顺序对应；
- 带图请求不与其它请求合并 prefill；图像 span 必须落在 prompt 内（Python 侧展开时保证）；
- 前缀缓存 / 多轮复用尚未实现，每个请求都会重新编码图像；
- 21 环境下服务模式若加载不到 HF tokenizer，会退回 fastllm 原生 tokenizer 编码 prompt（两者对占位符的 id 相同）。

## 数值验证

`test/basic/deepseek_v41_reference.py` 用官方 `inference/model.py` 的模块（把 tilelang kernel 换成纯 torch 实现）
构造随机初始化的迷你 V4.1 模型，与 FastLLM 逐步比较 logits，并可逐层比较中间张量：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_reference.py \
  --work-dir /tmp/v41-tiny --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \
  --experts 2 --activated 2 --index-topk 128 --candidate-topk-blocks 64 --no-fake-quant --regenerate
```

- `--experts 2 --activated 2` 与 `--index-topk 128` 消除随机权重下专家选择 / top-k 选择对微小数值差的敏感性，
  这时两侧 13 个贪心 token 完全一致、每步 logits 余弦相似度 >= 0.9998；
- 去掉 `--no-fake-quant` 可以验证与官方一致的伪量化路径（量化边界翻转会带来可见但有限的差异）；
- `--quant-format real` 会按真实 checkpoint 的格式（FP8 32x32、FP4 专家）保存迷你模型，用于验证加载器；
- `--dump-dir` 逐层比较隐藏状态、注意力输出、压缩 KV、indexer 分数与 top-k，并用 numpy 复算候选块 / top-k 选择。

图文输入用 `test/basic/deepseek_v41_vision_reference.py`（迷你文本模型 + 小规模视觉塔，图片由程序合成）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_vision_reference.py \
  --work-dir /tmp/v41-tiny-vision --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference --no-fake-quant --regenerate --dump-dir /tmp/v41-vision-dump
```

- 先比较 Python 预处理与官方 `image_processor.prepare_vl_inputs` 的 token 序列与 patch（要求完全一致），再逐步比较 logits；
  `--dump-dir` 时另外比较 ViT 各阶段、aligner 输出与合并后的输入嵌入（`--ref-vision-fp32` 让参考侧视觉塔用 float32，
  排除官方 bf16 路径的舍入噪声，此时 ViT / aligner 输出 cos = 1.000000）；
- `--experts 8 --bias-vl-boost 5` 给 `gate.bias_vl` 的最后两个专家加偏置，检查图像 token 的路由确实使用 `bias_vl`
  （随机权重下 8 专家的贪心 token 会因路由并列翻转而不同，属已知敏感性，正确性以 2 专家配置为准）；
DSpark 用 `test/basic/deepseek_v41_dspark.py`（迷你文本模型 + 3 个随机初始化的草稿层）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_dspark.py \
  --work-dir /tmp/v41-tiny-dspark --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference --regenerate
```

- 先跑一遍不开 DSpark 的基准（带 logits），再跑一遍开 DSpark 的，逐 token 比较并统计接受率；
- 用 `FASTLLM_DSPARK_FORCE_DRAFTS` 注入候选构造"接受 0 个 / 接受一部分 / 全部接受 / 混合"四种场景，
  校验每轮的实际接受长度与构造的一致、回滚后的续写与基准一致；
- 迷你模型的 logits 是 BF16（分辨率约 1/32），几乎并列的 argmax 会因 GEMM 选核不同而翻转，
  脚本把"差距在 3 个 BF16 ulp 以内"的分歧判为并列，以 DSpark 的输出为新前缀重新跑基准继续比较；
- 默认用 `--dtype float32` 与 2 专家 top-2、`index_topk` 大于压缩块数，减少随机权重下的并列。

- `--real-vision /path/to/DeepSeek-V4.1-Flash --image-size 640x480,1600x1200` 用真实 ViT 权重
  （aligner 维度依赖文本侧 dim，仍为随机）验证 32 层 ViT，包括接近 1024 token 上限的大图；
- `--chunked-prefill 16` 验证带图 prompt 的分块 prefill。

## 性能

稀疏注意力与 indexer 打分是 prefill 的两个主要开销，SM80 及以上的设备走 BF16 tensor core 路径：

- **稀疏注意力**（`V41SparseAttentionMmaKernel`）：一个 block 负责一个 token 的 32 个 head，
  候选（滑窗 + 压缩 top-k）按 32 个一组进共享内存，QK^T 与 PV 都用 `mma.sync.m16n8k16`
  （BF16 输入 / FP32 累加），片段用 `ldmatrix(.trans)` 装载，Q 常驻共享内存。
  在线 softmax 与 attention sink 的语义、候选顺序都与 CPU 参考实现一致，只有 FP32 累加顺序不同。
  decode 时 token 数少，按候选维再切成若干份（split-K），部分和由合并 kernel 按在线 softmax 的
  合并公式汇总，把 block 数补到约 4 倍 SM 数。
- **indexer 打分**（`V41IndexerScoreMmaKernel`）：一个 block 负责 64 个 token x 64 个候选；
  indexer 的 key 没有 head 维，K tile 只装载一次、循环 head 时只换 Q tile。
  整块落在因果可见范围外时直接写 `-inf` 跳过计算（这些位置本来就会被候选块打分与 top-k 忽略）。
- **indexer 分数矩阵的显存**：`[token, m]` 随上下文线性增长（1M 上下文的 ratio-1 层，
  4096 token 的分块要 16 GB）。现在按 token 维分块调用「打分 -> 候选块 -> top-k」，
  峰值由 `FASTLLM_DSV41_INDEX_SCORE_MB`（默认 128 MB）控制，与上下文长度解耦。
- **Hyper-Connections 混合系数**（`V41HcMixKernelMulti`）：`hcMult` 作为模板参数，
  累加器进寄存器；一个 block 处理 4 个相邻 token，复用同一份混合系数矩阵
  （真实模型是 24 x 20480 的 FP32，约 2 MB，原来每个 token 都要重读一遍）。
  结果与旧 kernel 逐 bit 相同。

3090 Ti（SM86）上用 `test/basic/deepseek_v41_reference.py --perf-config`
（4 层、64 头、head_dim 512、窗口 128、index_topk 512、32 个 indexer head）实测：

| 4096 token prefill + decode | 优化前 | 优化后 |
| --- | --- | --- |
| 稀疏注意力（4 层合计 / 每层 prefill） | 574 ms / 165 ms | 34 ms / 10.6 ms |
| indexer 打分（3 层合计） | 797 ms | 5.7 ms |
| HcMix（合计 / 每次 prefill 调用） | 23.3 ms / 1.92 ms | 6.9 ms / 254 us |
| 端到端 prefill | 1.56 s | 0.32 s |
| 端到端 decode | 102 tok/s | 271 tok/s |
| 单 token decode 的注意力 kernel | 88 us | 12 us（+ 6 us 合并） |

长上下文（同一配置，`--chunked-prefill 4096`）：

| prefill 长度 | 旧 kernel | 新 kernel | decode（旧 / 新） |
| --- | --- | --- | --- |
| 4096 | 1.56 s | 0.32 s | 102 / 271 tok/s |
| 8192 | 3.77 s | 0.39 s | 86 / 172 tok/s |
| 32768 | 33.60 s | 1.08 s | 58 / 99 tok/s |
| 65536 | — | 2.87 s | — |

indexer 分数矩阵的分块效果（65536 token prefill，扣掉同卡其它进程的 848 MiB 底噪）：
按 token 分块后峰值显存 1832 MiB，不分块（`FASTLLM_DSV41_INDEX_SCORE_MB` 设得很大）是 3770 MiB，
两者输出逐 token 相同，prefill 耗时 2.87 s vs 2.69 s（分块多约 7%）。

## 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_LEGACY_ATTN` | 稀疏注意力退回 FP32 标量 kernel（对比 / 排查用） |
| `FASTLLM_DSV41_LEGACY_INDEXER` | indexer 打分退回 FP32 标量 kernel |
| `FASTLLM_DSV41_LEGACY_HCMIX` | HC 混合系数退回旧 kernel |
| `FASTLLM_DSV41_ATTN_SPLITS` | 手动指定稀疏注意力候选维的 split-K 份数（默认自动） |
| `FASTLLM_DSV41_INDEX_SCORE_MB` | indexer 分数矩阵的显存预算（MB，默认 128），决定 token 维分块大小 |
| `FASTLLM_DSV41_INDEX_CHUNK` | 直接指定 indexer 的 token 分块大小（覆盖上面的预算推算） |
| `FASTLLM_DSV41_ENGRAM_META` | Engram 元数据 JSON 路径 |
| `FASTLLM_DSV41_ENGRAM_MMAP` | 以 mmap 方式访问 Engram 表 |
| `FASTLLM_DSV41_DISABLE_FAKE_QUANT` | 关闭 FP8 / FP4 伪量化（仅用于对齐调试） |
| `FASTLLM_DSV41_DISABLE_CUDA_ROUTE` | 路由退回 CPU 参考实现 |
| `FASTLLM_DSV41_DUMP_DIR` | 把每层中间张量写到该目录（对齐调试） |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` 等 | 前缀缓存相关，见"多请求与前缀缓存" |
| `FASTLLM_DSPARK_*` | DSpark 投机解码相关，见"DSpark 投机解码" |
