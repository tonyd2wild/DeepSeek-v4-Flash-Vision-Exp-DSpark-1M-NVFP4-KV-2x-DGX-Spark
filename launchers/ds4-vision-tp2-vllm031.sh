#!/usr/bin/env bash
# ds4-vision-tp2-vllm031.sh <0|1>
# DeepSeek-V4-Flash-Vision-Exp, TP2, on STOCK vLLM v0.31.0 (vllm/vllm-openai:v0.31.0).
# An alternative lane beside ds4-vision-tp2.sh; CURRENT.md is unchanged. Detail,
# measurements and the open items: docs/vllm-0.31-lane.md.
#
# What 0.31 brings upstream, so none of these are mounted here:
#   * Vision-Exp natively (DeepseekV4ForConditionalGeneration: ViT, aligner, bias_vl)
#     -- no ds4v_* files, no registry alias
#   * DSpark on model runner V2 (vllm#46995) -- no Patch 3 scheduler, no Patch 4 dspark.py
#   * the b12x MXFP4 MoE backend (--moe-backend b12x; W4A16 via VLLM_B12X_MOE_FP4_FORCE_A16)
#
# What this lane adds (all from this repo, mounted read-only):
#   * b12x==1.3.0 -- vLLM's OPTIONAL extra, absent from the stock image (preflight below)
#   * patches/safetensors_e8m0_plugin -- the eager loader cannot decode F8_E8M0 without it
#   * patches-031/dspark031_plugin -- LOAD-1: upstream's vision load_weights sorts the
#     whole checkpoint in memory, which OOMs under the eager loader
#   * patches-031/detokenizer.py -- Patch 5 (stop strings inside reasoning), ported
#
# Host defaults are ds4-vision-tp2.sh's; override any of them by environment.
set -euo pipefail
NODE_RANK="${1:?usage: ds4-vision-tp2-vllm031.sh <0|1>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
NAME="${NAME:-vllm_ds4_vision}"
MODEL_DIR="${MODEL_DIR:-DeepSeek-V4-Flash-Vision-Exp}"
MODEL_IN_CONTAINER="/models/$MODEL_DIR"
MASTER_ADDR="${MASTER_ADDR:-192.168.192.3}"
MASTER_PORT="${MASTER_PORT:-25440}"
PORT="${PORT:-8888}"
B12X_DIR="${B12X_DIR:-$HOME/.cache/vllm031-deps/b12x-1.3.0}"
# The KV pool in bytes, as measured on 2x GB10 (docs/vllm-0.31-lane.md). Set it empty
# to size by GPU_MEMORY_UTILIZATION instead (not yet tested on 0.31).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES-21789705851}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# --gpus all needs the NVIDIA container toolkit hook. Sandboxed hosts that refuse it
# (some rented Sparks) take: GPU_ARGS="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all"
GPU_ARGS="${GPU_ARGS:---gpus all}"
# RDMA devices for NCCL over RoCE; skipped when the host has none (NCCL then uses sockets).
IB_DEVICE="${IB_DEVICE:-/dev/infiniband}"
IB_ARGS=""; [ -e "$IB_DEVICE" ] && IB_ARGS="--device $IB_DEVICE:$IB_DEVICE"
# eager: required on a 64K-page kernel (the lazy loader hangs at shard 0 there). On
# stock DGX OS (4K pages) lazy works too; eager is what was measured.
LOAD_STRATEGY="${LOAD_STRATEGY:-eager}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
NCCL_IFACE="${NCCL_IFACE:-enp1s0f0np0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX-3}"   # set it EMPTY to let NCCL pick the GID
case "$NODE_RANK" in
  0) HOST_IP="${HOST_IP:-192.168.192.3}"; HEADLESS=""
     MODELS_HOST="${MODELS_HOST:-/mnt/bluey-models}" ;;
  1) HOST_IP="${HOST_IP:-192.168.192.1}"; HEADLESS="--headless"
     MODELS_HOST="${MODELS_HOST:-/var/tmp/models}" ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
test -d "$MODELS_HOST/$MODEL_DIR" || {
  echo "MODEL MISSING at $MODELS_HOST/$MODEL_DIR" >&2; exit 3; }
# b12x is vLLM's optional extra (vllm[b12x] -> b12x==1.3.0). Without it --moe-backend b12x
# fails at MoE init with "B12X_MXFP4_BF16 does not support ... current device".
test -d "$B12X_DIR/b12x" || {
  echo "b12x MISSING at $B12X_DIR. Install it beside the image, once per node:" >&2
  echo "  mkdir -p $(dirname "$B12X_DIR") && docker run --rm -v $(dirname "$B12X_DIR"):/out --entrypoint pip $IMAGE install -q --no-deps --target /out/$(basename "$B12X_DIR") b12x==1.3.0" >&2
  exit 4; }
# Its OWN JIT cache: 0.31's FlashInfer/DeepGEMM/Triton builds must not mix with 0.21's.
mkdir -p "$HOME/.cache/vllm-dspark031" "$HOME/.cache/huggingface"
docker rm -f "$NAME" 2>/dev/null || true
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
REASON='{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
SP=/usr/local/lib/python3.12/dist-packages
KVB=""; [ -n "$KV_CACHE_MEMORY_BYTES" ] && KVB="--kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES"
docker run -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  $GPU_ARGS $IB_ARGS \
  --entrypoint bash \
  -v "$MODELS_HOST:/models:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/vllm-dspark031:/vllm-cache" \
  -v "$B12X_DIR:/opt/b12x:ro" \
  -v "$REPO/patches-031/dspark031_plugin:/opt/dspark031:ro" \
  -v "$REPO/patches/safetensors_e8m0_plugin/safetensors_e8m0.py:$SP/safetensors_e8m0.py:ro" \
  -v "$REPO/patches/safetensors_e8m0_plugin/safetensors_e8m0-0.1.dist-info:$SP/safetensors_e8m0-0.1.dist-info:ro" \
  `# Patch 5 on 0.31: vllm#52486 plus the output-side guard (#42). patches-031/pristine has the source it derives from.` \
  -v "$REPO/patches-031/detokenizer.py:$SP/vllm/v1/engine/detokenizer.py:ro" \
  -e PYTHONPATH=/opt/dspark031:/opt/b12x \
  `# KV-7 (commit-time block dedupe) is off here: it is not part of CURRENT.md's recipe. 1 turns it on.` \
  -e DSPARK_BLOCK_DEDUPE="${DSPARK_BLOCK_DEDUPE:-0}" \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_CACHE_ROOT=/vllm-cache \
  -e DG_JIT_CACHE_DIR=/vllm-cache/deepgemm-cache \
  -e FLASHINFER_WORKSPACE_BASE=/vllm-cache/flashinfer \
  -e TILELANG_CACHE_DIR=/vllm-cache/tilelang \
  -e TORCHINDUCTOR_CACHE_DIR=/vllm-cache/torchinductor-cache \
  -e TRITON_CACHE_DIR=/vllm-cache/triton-cache \
  -e TORCH_EXTENSIONS_DIR=/vllm-cache/torch_extensions \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IFACE" -e GLOO_SOCKET_IFNAME="$NCCL_IFACE" -e TP_SOCKET_IFNAME="$NCCL_IFACE" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 \
  "$IMAGE" \
  -lc "
    exec vllm serve $MODEL_IN_CONTAINER \
      --hf-overrides '{\"architectures\":[\"DeepseekV4ForConditionalGeneration\"]}' \
      --served-model-name deepseek-v4-flash-dspark \
      --host 0.0.0.0 --port $PORT \
      --trust-remote-code \
      --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --kv-cache-dtype fp8 \
      --block-size 256 \
      --max-model-len 1048576 \
      --max-num-seqs 12 \
      --max-num-batched-tokens 8192 \
      --long-prefill-token-threshold 1024 \
      --compilation-config '{\"max_cudagraph_capture_size\":72}' \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION $KVB \
      --safetensors-load-strategy $LOAD_STRATEGY \
      --moe-backend b12x \
      --enable-prefix-caching \
      --async-scheduling \
      --enable-chunked-prefill \
      --speculative-config '$SPEC' \
      --tokenizer-mode deepseek_v4 \
      --distributed-executor-backend mp \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --reasoning-config '$REASON' \
      --default-chat-template-kwargs '{\"thinking\":false}' \
      --generation-config vllm \
      --enable-flashinfer-autotune \
      --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT $HEADLESS
  "
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP models=$MODELS_HOST image=$IMAGE load=$LOAD_STRATEGY rdma=${IB_ARGS:-none}"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited immediately" >&2; exit 1; }
