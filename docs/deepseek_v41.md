# DeepSeek-V4.1-Flash 支持说明

[返回 README](../README.md) · [DeepSeek-V4 部署指南](deepseek.md) · [混合推理](mixforward.md)

本文记录 FastLLM 对 DeepSeek-V4.1 系列（`model_type = deepseek_v41`，目前为 DeepSeek-V4.1-Flash）的支持范围、
架构差异、启动方式与验证方法。实现位于：

- `include/models/deepseekv41.h` / `src/models/deepseekv41.cpp`：模型（继承 `DeepSeekV4Model`，复用其 MoE / HC-post / WoA / 采样等基础设施）
- `src/models/deepseekv41_vision.cpp`：视觉编码器（ViT + aligner）与图文前向 `ForwardMultimodal`
- `src/devices/cpu/deepseekv41ops.cpp`：V4.1 专用算子的 CPU 参考实现
- `src/devices/cuda/models/deepseekv41-kernels.cu`：对应的 CUDA kernel（面向 SM86 等无 FP8 tensor core 的设备，FP32 计算）
- `tools/fastllm_pytools/deepseek_v41_engram.py`：Engram 哈希元数据生成
- `tools/fastllm_pytools/encoding_dsv41.py`：官方 V4.1 prompt 编码（vendored）
- `tools/fastllm_pytools/deepseek_v41_multimodal.py`：图像动态分辨率预处理（移植自官方 `image_processor.py`）与图像占位符展开
- `test/basic/deepseek_v41_reference.py`：与官方 `inference/model.py` 的端到端数值对齐测试
- `test/basic/deepseek_v41_vision_reference.py`：图文输入的端到端对齐测试（含真实 ViT 权重验证）
- `test/basic/test_deepseek_v41_cpu_fixture.py`：CPU-only 的固定 fixture 对齐回归（只依赖 numpy，可进 CI）
- `test/ops/deepseekV41OpsRegression.cpp`：11 个 V4.1 专用算子的 CPU / CUDA 一致性测试

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
| 附加模块 | MTP / DSpark | DSpark 草稿层（mtp.*）、视觉编码器（vision.* / aligner.* / image_start / image_end / image_newline） |

## 当前支持范围

已实现并通过数值对齐测试：

- 文本推理（prefill + decode，含分块 prefill），CUDA 与 CPU 两套算子路径；
- 多请求批量 decode：多个请求的 token 拼成一个序列共享一次前向（Linear / MoE / Engram 查表按整批执行，
  注意力与压缩 KV 按请求分别执行），见下文"多请求与前缀缓存"；
- 前缀缓存 / 多轮对话复用（`--cache_history true`）；
- 跨层共享压缩 KV（ratio 1 / 2）、两级 indexer top-k、Engram、Hyper-Connections、sqrt-softplus 路由；
- 与官方实现一致的 FP8 / FP4 伪量化（窗口 KV、压缩 KV、indexer q/k）；
- 真实 checkpoint 的权重格式（FP8 32x32、FP4 路由专家、FP8 + UE8M0 的 Engram 表）；
- 图像输入（OpenAI 接口的 `image_url`）：ViT + aligner、图像 token 的 `gate.bias_vl` 路由与 Engram 掩码，详见下文。

尚未实现：

- DSpark 投机解码（`mtp.*` 权重不加载）；
- CUDA Graph、张量并行等 V4 已有的性能特性。

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
- `--kv_cache_dtype fp8_e4m3` 让滑窗 KV 与压缩 KV 以 FP8 E4M3 + UE8M0 块 scale（每 32 个一组）存储，
  每行 528 B（BF16 为 1024 B）。滑窗 KV 本身就在 FP8 网格上，存储无损；压缩 KV 为 FP4 网格，FP8 存储带来
  不超过 2^-4 的相对舍入（迷你模型上开启伪量化时，与 BF16 存储的逐步 cos 相当）。默认仍为 BF16；
- `ftllm` launcher 的自动配置按 config 计算 V4.1 的常驻内存下界（FP4 专家约 289 GB + Engram 表约 203 GB +
  稠密部分），权重文件不全时也不会低估；主机内存不足以放下专家与 Engram 表时会退到 `moe_device=disk`；
- 单路 CPU 机器可用 `--moe_device cpu`；
- 内存需求：Engram 表约 200 GB + 路由专家（FP4）约 270 GB + 加载临时空间；
- 首次启动会生成 `engram_meta.json`（约 1 分钟）并读入两张 Engram 表。

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
- `--real-vision /path/to/DeepSeek-V4.1-Flash --image-size 640x480,1600x1200` 用真实 ViT 权重
  （aligner 维度依赖文本侧 dim，仍为随机）验证 32 层 ViT，包括接近 1024 token 上限的大图；
- `--chunked-prefill 16` 验证带图 prompt 的分块 prefill。

### CPU-only 固定 fixture 回归（无需 torch）

`test/basic/deepseek_v41_reference.py` 需要 torch、transformers 与官方 `inference/` 代码，进不了 CI。
`test/basic/test_deepseek_v41_cpu_fixture.py` 把「一个微型 V4.1 checkpoint + 官方实现算出的参考 logits」
固化在 `test/basic/deepseek_v41_fixture.npz`（约 2.2 MB）里，只依赖 numpy 与 fastllm 的 Python 包：

```bash
PYTHONPATH=build/tools python test/basic/test_deepseek_v41_cpu_fixture.py
```

退出码 0 表示通过，1 表示失败；fixture 缺失时打印重新生成的命令并以 0 退出（跳过）。
fixture 里的模型是 5 层、dim 128、词表 128、2 专家，覆盖 V4.1 的全部结构特性：
`compress_ratios = (0, 2, 2, 1, 1)`（三种压缩层）、`kv_source_layer_ids = (1, 3)`（层 2 / 4 跨层复用压缩 KV）、
`index_source_layer_ids = (1, 3, 4)` 且 `candidate_source_layer_id = 3`（两级 top-k）、
`engram_layer_ids = (1, 4)`、`hc_mult = 2`、`window_size = 8`（48 token 的 prefill 让滑窗环形缓冲绕多圈）。

比较方式：逐步（prefill + 4 步 decode）比较 logits 的余弦相似度（>= 0.9975）与
`max|diff|` 占 logits 值域的比例（<= 4%），并要求贪心 token 与参考一致；
参考侧是官方实现的 bf16 前向，噪声底约为值域的 2–3%（fastllm 自己的 CPU 与 CUDA 路径之间也有约 1.5%），
所以容差按值域折算而不是给绝对值。fixture 生成时会在多个提示词种子里挑一个每步 top1/top2 间距都
远大于该噪声的，避免 argmax 因舍入翻转；万一仍然翻转，测试会按"参考侧两者间距落在容差内"判为并列而不算失败。

重新生成 fixture（需要 torch + 官方 inference 代码 + 一张 GPU）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_fixture_gen.py \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \
  --work-dir /tmp/v41-micro --out test/basic/deepseek_v41_fixture.npz
```

生成脚本自带一个 128 词的迷你 tokenizer，不依赖真实 checkpoint 的 tokenizer.json。

### 算子级 CPU / CUDA 一致性

`test/ops/deepseekV41OpsRegression.cpp`（`cmake -DUNIT_TEST=ON`）用同一份输入分别在 CPU 与 CUDA 上跑
11 个 V4.1 专用算子并比较输出，覆盖随机输入与边界情况：

```bash
./build/deepseekV41OpsRegression            # 全部算子
./build/deepseekV41OpsRegression --list     # 列出用例
./build/deepseekV41OpsRegression IndexerTopK CandidateBlocks   # 只跑指定算子
ctest -R deepseekV41Ops                     # 无 CUDA 设备时以 77 跳过
```

## 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_ENGRAM_META` | Engram 元数据 JSON 路径 |
| `FASTLLM_DSV41_ENGRAM_MMAP` | 以 mmap 方式访问 Engram 表 |
| `FASTLLM_DSV41_DISABLE_FAKE_QUANT` | 关闭 FP8 / FP4 伪量化（仅用于对齐调试） |
| `FASTLLM_DSV41_DISABLE_CUDA_ROUTE` | 路由退回 CPU 参考实现 |
| `FASTLLM_DSV41_DUMP_DIR` | 把每层中间张量写到该目录（对齐调试） |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` 等 | 前缀缓存相关，见"多请求与前缀缓存" |
