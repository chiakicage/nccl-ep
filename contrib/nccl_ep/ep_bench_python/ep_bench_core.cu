/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Throughput and validation methodology aligned with DeepEP (https://github.com/deepseek-ai/DeepEP).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cupti.h>
#include <nvtx3/nvToolsExt.h>
#include <nccl.h>
#include <nccl_device.h>
#include "nccl_ep.h"
#include "ep_bench_core.h"

static void failWithMessage(const char* kind, const char* file, int line, const char* message) {
  char buffer[1024];
  snprintf(buffer, sizeof(buffer), "Failed: %s error %s:%d '%s'", kind, file, line, message);
  throw std::runtime_error(buffer);
}

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    failWithMessage("Cuda", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    failWithMessage("NCCL", __FILE__, __LINE__, ncclGetErrorString(r)); \
  }                                                 \
} while(0)

// ============================================================================
// KernelTimer: CUPTI Activity API-based per-kernel GPU timing
// ============================================================================
// Records per-kernel GPU execution times by matching kernel name substrings.
// Entirely benchmark-side — zero impact on the production nccl_ep library.
// Same mechanism used by PyTorch kineto (torch.profiler).

#define CUPTI_CALL(call) do {                                                  \
    CUptiResult _s = (call);                                                   \
    if (_s != CUPTI_SUCCESS) {                                                 \
        const char* _e; cuptiGetResultString(_s, &_e);                        \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", __FILE__, __LINE__, _e);   \
    }                                                                          \
} while (0)

static const size_t CUPTI_BUF_SIZE = 8 * 1024 * 1024;  // 8 MB per buffer

struct KernelStat { uint64_t total_ns = 0; int count = 0; };
// Global accumulator populated by CUPTI buffer-completed callback
static std::map<std::string, KernelStat> g_kernel_stats;

static void CUPTIAPI cuptiBufferRequested(uint8_t** buf, size_t* sz, size_t* maxRecords) {
    // aligned_alloc requires size to be a multiple of alignment
    *buf = static_cast<uint8_t*>(aligned_alloc(8, CUPTI_BUF_SIZE));
    *sz = CUPTI_BUF_SIZE;
    *maxRecords = 0;
}

static void CUPTIAPI cuptiBufferCompleted(CUcontext /*ctx*/, uint32_t /*streamId*/,
                                           uint8_t* buf, size_t /*sz*/, size_t validSz) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buf, validSz, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = reinterpret_cast<CUpti_ActivityKernel5*>(record);
            if (k->name) {
                g_kernel_stats[k->name].total_ns += k->end - k->start;
                g_kernel_stats[k->name].count++;
            }
        }
    }
    free(buf);
}

class KernelTimer {
public:
    // Enable CUPTI kernel activity recording and clear accumulated stats.
    void start() {
        g_kernel_stats.clear();
        CUPTI_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Flush all pending CUPTI buffers and disable recording.
    void stop() {
        CUPTI_CALL(cuptiActivityFlushAll(0));
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Average GPU execution time (microseconds) across all kernels whose
    // mangled name contains substr.  Returns 0 if no matching kernel found.
    double get_avg_us(const char* substr) const {
        uint64_t total_ns = 0; int count = 0;
        for (const auto& kv : g_kernel_stats) {
            if (kv.first.find(substr) != std::string::npos) {
                total_ns += kv.second.total_ns;
                count    += kv.second.count;
            }
        }
        return count ? static_cast<double>(total_ns) / count / 1000.0 : 0.0;
    }

    // Print all captured kernel names and their stats to stdout (debug helper).
    void dump(int rank) const {
        if (rank != 0) return;
        printf("[KernelTimer] Captured %zu distinct kernel(s):\n", g_kernel_stats.size());
        for (const auto& kv : g_kernel_stats) {
            double avg_us = static_cast<double>(kv.second.total_ns) / kv.second.count / 1000.0;
            printf("  count=%3d  avg=%.2f us  %s\n", kv.second.count, avg_us, kv.first.c_str());
        }
        fflush(stdout);
    }
};

// CUDA allocator callbacks for ncclEpCreateGroup
// These are used by ncclEpTensorCreate/Destroy to allocate/free tensor memory
static cudaError_t cudaAllocCallback(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

static cudaError_t cudaFreeCallback(void* ptr) {
    return cudaFree(ptr);
}

static void ncclBarrier(ncclComm_t comm, cudaStream_t stream, void* workspace = nullptr) {
    int* barrier_value = nullptr;
    bool owns_workspace = false;
    if (workspace) {
        barrier_value = static_cast<int*>(workspace);
    } else {
        CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&barrier_value), sizeof(int)));
        owns_workspace = true;
    }
    CUDACHECK(cudaMemset(barrier_value, 0, sizeof(int)));
    NCCLCHECK(ncclAllReduce(barrier_value, barrier_value, 1, ncclInt, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    if (owns_workspace) {
        CUDACHECK(cudaFree(barrier_value));
    }
}

static double elapsedMs(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Structure to hold all tensors needed for benchmarking
struct BenchmarkTensors {
    // Dispatch tensors
    ncclNDTensor_t inputs[3];
    ncclNDTensor_t outputs[3];
    ncclNDTensor_t local_tensors[1];
    int num_dispatch_inputs;
    int num_dispatch_outputs;

    // Combine tensors
    ncclNDTensor_t combine_inputs[2];
    ncclNDTensor_t combine_outputs[2];
    ncclNDTensor_t combine_local_tensors[1];
    int num_combine_inputs;
    int num_combine_outputs;
    int num_combine_local_tensors;

    // Owned tensors (for cleanup)
    ncclNDTensor_t dispatch_topk_weights;
    ncclNDTensor_t expert_outputs;
    ncclNDTensor_t combined_output;
    ncclNDTensor_t topk_weights;
    ncclNDTensor_t combine_output_topk_weights;

    bool is_ll_mode;
};

// Setup tensors for LOW_LATENCY mode using ncclEpTensorCreate
void setupLowLatencyTensors(
    ncclEpGroup_t ep_group,
    BenchmarkTensors& tensors,
    ncclNDTensor_t& topk_idx,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int max_tokens_per_rank,
    int nRanks
) {
    tensors.is_ll_mode = true;
    tensors.num_dispatch_inputs = 1;
    tensors.num_dispatch_outputs = 1;
    tensors.num_combine_inputs = 1;
    tensors.num_combine_outputs = 1;
    tensors.num_combine_local_tensors = 1;

    // Dispatch input: tokens
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.inputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // Dispatch output: 3D [num_local_experts, max_tokens_per_rank * nRanks, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[0], 3, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_local_experts, max_tokens_per_rank * nRanks, hidden));

    // Local tensors: recv expert counter (device memory) - required for dispatch
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.local_tensors[0], 1, ncclInt32,
                                   NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                                   nullptr, num_local_experts));

    // Combine input: 3D expert outputs
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.expert_outputs, 3, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_local_experts, max_tokens_per_rank * nRanks, hidden));

    // Combine output
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combined_output, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // topk_weights as local tensor for combine
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Setup combine arrays
    tensors.combine_inputs[0] = tensors.expert_outputs;
    tensors.combine_outputs[0] = tensors.combined_output;
    tensors.combine_local_tensors[0] = tensors.topk_weights;
}

// Setup tensors for HIGH_THROUGHPUT mode using ncclEpTensorCreate
void setupHighThroughputTensors(
    ncclEpGroup_t ep_group,
    BenchmarkTensors& tensors,
    ncclNDTensor_t& topk_idx,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int num_recv_tokens
) {
    tensors.is_ll_mode = false;
    tensors.num_dispatch_inputs = 3;
    tensors.num_dispatch_outputs = 3;
    // HT combine uses only 1 input (expert_outputs) and 1 output (combined_output)
    tensors.num_combine_inputs = 1;
    tensors.num_combine_outputs = 1;
    tensors.num_combine_local_tensors = 0;

    // Dispatch input: tokens - initialize with test pattern
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.inputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));
    {
        void* input0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.inputs[0], &input0_data));
        CUDACHECK(cudaMemset(input0_data, 0, num_tokens * hidden * 2));
    }

    // Dispatch input: topk_weights - initialize with equal weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.dispatch_topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));
    {
        float *topk_weights_host = new float[num_tokens * top_k];
        for (unsigned int i = 0; i < num_tokens * top_k; i++) {
            topk_weights_host[i] = 1.0f / top_k;
        }
        void* dtw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.dispatch_topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
        delete[] topk_weights_host;
    }
    tensors.inputs[1] = tensors.dispatch_topk_weights;

    tensors.inputs[2] = topk_idx;

    // Dispatch output: 2D [num_recv_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[0], 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_recv_tokens, hidden));

    // Dispatch output: recv_topk_weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[1], 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_recv_tokens, top_k));

    // Dispatch output: recv_topk_idx
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.outputs[2], 2, ncclInt64,
                                   NCCL_EP_TENSOR_TAG_TOPK_IDX,
                                   nullptr, num_recv_tokens, top_k));

    // Local tensors: recv expert counter (device memory) - required for dispatch
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.local_tensors[0], 1, ncclInt32,
                                   NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                                   nullptr, num_local_experts));

    // Combine input: 2D expert outputs - same size as dispatch output (received token count)
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.expert_outputs, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_recv_tokens, hidden));
    {
        void* eo_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.expert_outputs, &eo_data));
        CUDACHECK(cudaMemset(eo_data, 0, num_recv_tokens * hidden * 2));
    }

    // Combine output - sized to num_tokens (original token count per rank)
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combined_output, 2, ncclBfloat16,
                                   NCCL_EP_TENSOR_TAG_TOKENS,
                                   nullptr, num_tokens, hidden));

    // topk_weights as regular input for combine
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Combine output: topk_weights
    NCCLCHECK(ncclEpTensorCreate(ep_group, &tensors.combine_output_topk_weights, 2, ncclFloat32,
                                   NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                                   nullptr, num_tokens, top_k));

    // Setup combine arrays
    tensors.combine_inputs[0] = tensors.expert_outputs;
    tensors.combine_inputs[1] = tensors.topk_weights;
    tensors.combine_outputs[0] = tensors.combined_output;
    tensors.combine_outputs[1] = tensors.combine_output_topk_weights;
}

// Cleanup benchmark tensors using ncclEpTensorDestroy
void cleanupBenchmarkTensors(ncclEpGroup_t ep_group, BenchmarkTensors& tensors, ncclNDTensor_t topk_idx) {
    // topk_idx is created with ncclEpTensorCreate (user-provided data_ptr)
    {
        void* topk_data;
        ncclEpTensorGetData(topk_idx, &topk_data);
        if (topk_data) cudaFree(topk_data);
        ncclEpTensorDestroy(ep_group, topk_idx);
    }

    // All other tensors are created with ncclEpTensorCreate
    ncclEpTensorDestroy(ep_group, tensors.inputs[0]);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.dispatch_topk_weights);
    }

    ncclEpTensorDestroy(ep_group, tensors.outputs[0]);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.outputs[1]);
        ncclEpTensorDestroy(ep_group, tensors.outputs[2]);
    }

    ncclEpTensorDestroy(ep_group, tensors.local_tensors[0]);
    ncclEpTensorDestroy(ep_group, tensors.expert_outputs);
    ncclEpTensorDestroy(ep_group, tensors.combined_output);
    ncclEpTensorDestroy(ep_group, tensors.topk_weights);

    if (!tensors.is_ll_mode) {
        ncclEpTensorDestroy(ep_group, tensors.combine_output_topk_weights);
    }
}

// ============================================================================
// Data Validation Support (similar to DeepEP test_internode.py / test_low_latency.py)
//
// Methodology:
//   - Input tokens are fingerprinted with (source_rank, token_id) in BF16.
//   - Dispatch validation recomputes expected routing deterministically and
//     verifies each received token's identity and integrity.
//   - Combine validation computes expected weighted sums analytically and
//     compares against actual output using a cosine-similarity metric (calc_diff).
// ============================================================================

// Rank offset for BF16 precision: integers > 256 lose precision in BF16
// Using negative values (rank - 128) allows up to 256 ranks
static const int RANK_OFFSET = 128;

// Number of columns to embed token index (for full traceability)
// Matches DeepEP's approach: last 128 columns store token index
static const int TOKEN_ID_COLS = 128;

// Helper: Convert BF16 to float (CPU-side)
static float bf16ToFloat(uint16_t bf16) {
    uint32_t bits = (static_cast<uint32_t>(bf16)) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Helper: Convert float to BF16 (CPU-side, truncation — used only for initialization)
static uint16_t floatToBf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Cosine-similarity-based discrepancy metric in double precision
// Returns 0 for perfect match, larger values for worse match
static double calc_diff(const double* x, const double* y, size_t n) {
    double dot_xy = 0, dot_xx = 0, dot_yy = 0;
    for (size_t i = 0; i < n; i++) {
        double xi = x[i] + 1.0;
        double yi = y[i] + 1.0;
        dot_xy += xi * yi;
        dot_xx += xi * xi;
        dot_yy += yi * yi;
    }
    double denom = dot_xx + dot_yy;
    if (denom == 0) return 0;
    return 1.0 - 2.0 * dot_xy / denom;
}

// Initialize input tensors with validation-friendly patterns (DeepEP style)
// Pattern: each element = (rank - RANK_OFFSET) except last TOKEN_ID_COLS columns = token_index
void initializeValidationData(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    int myRank,
    bool is_ht_mode
) {
    // Calculate the rank value to use (handles BF16 precision limits)
    float rank_value = static_cast<float>(myRank - RANK_OFFSET);
    uint16_t rank_bf16 = floatToBf16(rank_value);

    // Allocate host buffer for token data
    size_t token_size = num_tokens * hidden;
    uint16_t* token_data_host = new uint16_t[token_size];

    // Fill token data with rank value, embed token index in last TOKEN_ID_COLS columns.
    // Token ID is split into high (t/256) and low (t%256) bytes to stay within BF16's
    // exact integer range (0-255). First TOKEN_ID column = high byte, rest = low byte.
    for (unsigned int t = 0; t < num_tokens; t++) {
        uint16_t token_hi = floatToBf16(static_cast<float>(t / 256));
        uint16_t token_lo = floatToBf16(static_cast<float>(t % 256));
        for (unsigned int h = 0; h < hidden; h++) {
            if (h == hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_hi;
            } else if (h > hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_lo;
            } else {
                token_data_host[t * hidden + h] = rank_bf16;
            }
        }
    }

    // Copy to GPU
    {
        void* input0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.inputs[0], &input0_data));
        CUDACHECK(cudaMemcpy(input0_data, token_data_host,
                             token_size * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }

    // Generate random positive topk_weights: abs(randn)
    // LL: weights applied during combine → affects combined output
    // HT: weights forwarded during dispatch → does NOT affect combined output
    float* topk_weights_host = new float[num_tokens * top_k];
    std::mt19937 rng(42 + myRank);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        topk_weights_host[i] = std::abs(normal(rng));
        if (topk_weights_host[i] < 1e-6f) topk_weights_host[i] = 1e-6f;
    }

    if (is_ht_mode) {
        void* dtw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.dispatch_topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Also initialize the combine topk_weights (used by both modes)
    {
        void* tw_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.topk_weights, &tw_data));
        CUDACHECK(cudaMemcpy(tw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }

    delete[] topk_weights_host;
    delete[] token_data_host;
}

// Validation result structure
struct ValidationResult {
    bool passed;
    int errors;
    double max_diff;
    std::string message;
};

// Forward declaration (defined later in the file)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host, unsigned int num_tokens, unsigned int num_experts,
    unsigned int top_k, int rank, int seed = 1);

// Generate HT topk_idx for a given rank (deterministic)
// Randperm routing (uniform), consistent with Hybrid-EP (test_hybrid_ep.py)
static void generateTopkIndicesHT(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank
) {
    std::mt19937 gen(rank + 42);
    std::vector<int64_t> expert_perm(num_experts);
    std::iota(expert_perm.begin(), expert_perm.end(), 0);
    for (unsigned int i = 0; i < num_tokens; i++) {
        std::shuffle(expert_perm.begin(), expert_perm.end(), gen);
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = expert_perm[j];
        }
    }
}

// Extract (source_rank, token_id) from a received token row using first and last columns
static bool extractTokenIdentity(
    const uint16_t* row,
    unsigned int hidden,
    int nRanks,
    unsigned int num_tokens,
    int* out_source_rank,
    int* out_token_id
) {
    float rank_val = bf16ToFloat(row[0]);
    *out_source_rank = static_cast<int>(rank_val + RANK_OFFSET + 0.5f);

    float token_hi = bf16ToFloat(row[hidden - TOKEN_ID_COLS]);
    float token_lo = bf16ToFloat(row[hidden - 1]);
    *out_token_id = static_cast<int>(token_hi + 0.5f) * 256 + static_cast<int>(token_lo + 0.5f);

    return (*out_source_rank >= 0 && *out_source_rank < nRanks &&
            *out_token_id >= 0 && *out_token_id < static_cast<int>(num_tokens));
}

// Verify a received token row has consistent data (all rank cols match, all token_id cols match)
static bool verifyTokenIntegrity(
    const uint16_t* row,
    unsigned int hidden
) {
    uint16_t expected_rank_bf16 = row[0];
    for (unsigned int h = 1; h < hidden - TOKEN_ID_COLS; h++) {
        if (row[h] != expected_rank_bf16) return false;
    }
    // First TOKEN_ID column is the high byte (standalone), rest are low byte
    uint16_t expected_token_lo_bf16 = row[hidden - 1];
    for (unsigned int h = hidden - TOKEN_ID_COLS + 1; h < hidden - 1; h++) {
        if (row[h] != expected_token_lo_bf16) return false;
    }
    return true;
}

// Validate dispatch output: verify that expected tokens arrived at the correct experts.
// Recomputes every rank's topk_idx deterministically to build the expected set,
// then checks the dispatch output for missing, unexpected, or corrupted tokens.
ValidationResult validateDispatchOutput(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode
) {
    ValidationResult result = {true, 0, 0.0, ""};
    int errors = 0;
    const int max_errors_to_print = 10;
    int errors_printed = 0;

    // Temp buffer to recompute each rank's topk_idx
    int64_t* src_topk = new int64_t[num_tokens * top_k];

    if (!is_ht_mode) {
        // ==================== LL Mode ====================
        // Output: 3D [num_local_experts, max_tokens_per_expert, hidden]

        const unsigned int* out0_sizes; unsigned int out0_ndim;
        NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes, &out0_ndim));
        unsigned int max_tpe = out0_sizes[1];
        size_t total_size = static_cast<size_t>(num_local_experts) * max_tpe * hidden;
        uint16_t* recv_data = new uint16_t[total_size];
        void* output0_data_ll;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data_ll));
        CUDACHECK(cudaMemcpy(recv_data, output0_data_ll,
                             total_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        int* tokens_per_expert = new int[num_local_experts];
        void* local0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.local_tensors[0], &local0_data));
        CUDACHECK(cudaMemcpy(tokens_per_expert, local0_data,
                             num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));

        // Build expected set: expected[local_expert] = set of (source_rank, token_id)
        std::vector<std::set<std::pair<int,int>>> expected(num_local_experts);

        for (int r = 0; r < nRanks; r++) {
            generateRandomTopkIndicesLL(src_topk, num_tokens, num_experts, top_k, r);
            for (unsigned int t = 0; t < num_tokens; t++) {
                for (unsigned int k = 0; k < top_k; k++) {
                    int64_t expert_id = src_topk[t * top_k + k];
                    if (expert_id < 0) continue;
                    int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                    int local_expert = static_cast<int>(expert_id) % static_cast<int>(num_local_experts);
                    if (expert_rank == myRank) {
                        expected[local_expert].insert({r, static_cast<int>(t)});
                    }
                }
            }
        }

        // Scan output and match against expected
        for (unsigned int e = 0; e < num_local_experts; e++) {
            int count = tokens_per_expert[e];
            if (count < 0 || count > static_cast<int>(max_tpe)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] LL dispatch: expert %u has invalid count %d (max %u)\n",
                           myRank, e, count, max_tpe);
                    errors_printed++;
                }
                errors++;
                continue;
            }

            std::set<std::pair<int,int>> found;

            for (int j = 0; j < count; j++) {
                const uint16_t* row = recv_data + (e * max_tpe + j) * hidden;
                int source_rank = -1, token_id = -1;

                if (!extractTokenIdentity(row, hidden, nRanks, num_tokens, &source_rank, &token_id)) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: invalid identity (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                    continue;
                }

                if (!verifyTokenIntegrity(row, hidden)) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: data corruption (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                }

                auto key = std::make_pair(source_rank, token_id);
                if (expected[e].find(key) == expected[e].end()) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u slot %d: unexpected token (rank=%d, token=%d)\n",
                               myRank, e, j, source_rank, token_id);
                        errors_printed++;
                    }
                    errors++;
                }
                found.insert(key);
            }

            // Check for missing tokens
            for (const auto& key : expected[e]) {
                if (found.find(key) == found.end()) {
                    if (errors_printed < max_errors_to_print) {
                        printf("[Rank %d] LL dispatch: expert %u: missing token (rank=%d, token=%d)\n",
                               myRank, e, key.first, key.second);
                        errors_printed++;
                    }
                    errors++;
                }
            }
        }

        delete[] tokens_per_expert;
        delete[] recv_data;

    } else {
        // ==================== HT Mode ====================
        // FIXME: ncclEpHandleGetNumRecvTokens returns buffer max, not actual count — scan recv_topk_idx as workaround.
        // Output buffer is [nRanks * max_tokens_per_rank, hidden], tokens packed contiguously at 0..N-1.
        // We use recv_topk_idx (outputs[2]) to identify valid rows (expert index >= 0).

        const unsigned int* out0_sizes_ht; unsigned int out0_ndim_ht;
        NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes_ht, &out0_ndim_ht));
        unsigned int buf_rows = out0_sizes_ht[0];
        size_t recv_size = static_cast<size_t>(buf_rows) * hidden;
        uint16_t* recv_data = new uint16_t[recv_size];
        void* output0_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data));
        CUDACHECK(cudaMemcpy(recv_data, output0_data,
                             recv_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        // Read recv_topk_idx to identify valid rows
        int64_t* recv_topk_idx = new int64_t[static_cast<size_t>(buf_rows) * top_k];
        void* output2_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.outputs[2], &output2_data));
        CUDACHECK(cudaMemcpy(recv_topk_idx, output2_data,
                             static_cast<size_t>(buf_rows) * top_k * sizeof(int64_t),
                             cudaMemcpyDeviceToHost));

        // Build expected set from deterministic routing
        std::set<std::pair<int,int>> expected;
        for (int r = 0; r < nRanks; r++) {
            generateTopkIndicesHT(src_topk, num_tokens, num_experts, top_k, r);
            for (unsigned int t = 0; t < num_tokens; t++) {
                for (unsigned int k = 0; k < top_k; k++) {
                    int64_t expert_id = src_topk[t * top_k + k];
                    int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                    if (expert_rank == myRank) {
                        expected.insert({r, static_cast<int>(t)});
                        break;
                    }
                }
            }
        }

        // Scan ALL rows, but only validate rows where recv_topk_idx has valid entries
        std::set<std::pair<int,int>> found;

        for (unsigned int j = 0; j < buf_rows; j++) {
            // Check if this row has any valid expert index
            bool has_valid_expert = false;
            for (unsigned int k = 0; k < top_k; k++) {
                if (recv_topk_idx[j * top_k + k] >= 0) {
                    has_valid_expert = true;
                    break;
                }
            }
            if (!has_valid_expert) continue;

            const uint16_t* row = recv_data + j * hidden;
            int source_rank = -1, token_id = -1;

            if (!extractTokenIdentity(row, hidden, nRanks, num_tokens, &source_rank, &token_id)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: invalid identity (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
                continue;
            }

            if (!verifyTokenIntegrity(row, hidden)) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: data corruption (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
            }

            auto key = std::make_pair(source_rank, token_id);
            if (expected.find(key) == expected.end()) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: slot %u: unexpected token (rank=%d, token=%d)\n",
                           myRank, j, source_rank, token_id);
                    errors_printed++;
                }
                errors++;
            }
            found.insert(key);
        }

        // Check for missing tokens
        for (const auto& key : expected) {
            if (found.find(key) == found.end()) {
                if (errors_printed < max_errors_to_print) {
                    printf("[Rank %d] HT dispatch: missing token (rank=%d, token=%d)\n",
                           myRank, key.first, key.second);
                    errors_printed++;
                }
                errors++;
            }
        }

        delete[] recv_topk_idx;
        delete[] recv_data;
    }

    delete[] src_topk;

    result.errors = errors;
    result.passed = (errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s dispatch validation: %d errors",
                 is_ht_mode ? "HT" : "LL", errors);
        result.message = buf;
    }

    return result;
}

// Compute is_token_in_rank.sum() - count of unique ranks each token is sent to
// This matches DeepEP's validation approach
// Returns array of size num_tokens with unique rank count per token
int* countUniqueRanksPerToken(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int nRanks
) {
    int* unique_ranks = new int[num_tokens]();  // Zero-initialized
    unsigned int num_local_experts = num_experts / nRanks;

    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> ranks_set;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id >= 0) {
                int target_rank = expert_id / num_local_experts;
                ranks_set.insert(target_rank);
            }
        }
        unique_ranks[t] = ranks_set.size();
    }
    return unique_ranks;
}

// Count valid experts for each token (experts with topk_idx >= 0)
int countValidExperts(const int64_t* topk_idx_host, unsigned int token_idx, unsigned int top_k) {
    int count = 0;
    for (unsigned int k = 0; k < top_k; k++) {
        if (topk_idx_host[token_idx * top_k + k] >= 0) {
            count++;
        }
    }
    return count;
}

// Validate combine output for Low Latency mode
// DeepEP formula: check = combined / is_token_in_rank.sum()
// LL combine applies weighted sum: combined[t] = x[t] * sum(valid weights)
// Compared using calc_diff in double precision with threshold 1e-5
ValidationResult validateCombineOutputLL(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host
) {
    (void)num_experts;
    (void)nRanks;

    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.combined_output, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    float* topk_weights_host = new float[num_tokens * top_k];
    void* tw_data_ll;
    NCCLCHECK(ncclEpTensorGetData(tensors.topk_weights, &tw_data_ll));
    CUDACHECK(cudaMemcpy(topk_weights_host, tw_data_ll,
                         num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        if (countValidExperts(topk_idx_host, t, top_k) > 0)
            num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nv = countValidExperts(topk_idx_host, t, top_k);
        if (nv == 0) continue;

        double weight_sum = 0;
        for (unsigned int k = 0; k < top_k; k++) {
            if (topk_idx_host[t * top_k + k] >= 0)
                weight_sum += static_cast<double>(topk_weights_host[t * top_k + k]);
        }

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * weight_sum;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 1e-5) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "LL combine: calc_diff=%.6e (threshold=1e-5)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] topk_weights_host;
    delete[] combined_data;
    return result;
}

// Validate combine output for High Throughput mode
// DeepEP formula: check = combined / is_token_in_rank.sum()
// HT combine is unweighted sum: combined[t] = x[t] * num_unique_ranks
// Compared using calc_diff in double precision with threshold 5e-6
ValidationResult validateCombineOutputHT(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host
) {
    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(ncclEpTensorGetData(tensors.combined_output, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    int* unique_ranks = countUniqueRanksPerToken(topk_idx_host, num_tokens,
                                                  num_experts, top_k, nRanks);

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        if (unique_ranks[t] > 0) num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nr = unique_ranks[t];
        if (nr == 0) continue;

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));
        double scale = static_cast<double>(nr);

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * scale;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 5e-6) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "HT combine: calc_diff=%.6e (threshold=5e-6)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] unique_ranks;
    delete[] combined_data;
    return result;
}

// Wrapper that calls appropriate validation based on mode
ValidationResult validateCombineOutput(
    BenchmarkTensors& tensors,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode,
    int64_t* topk_idx_host
) {
    if (is_ht_mode) {
        return validateCombineOutputHT(tensors, num_tokens, hidden, num_experts,
                                        top_k, myRank, nRanks, topk_idx_host);
    } else {
        return validateCombineOutputLL(tensors, num_tokens, hidden, num_experts,
                                        top_k, myRank, nRanks, topk_idx_host);
    }
}

// Benchmark result structure
struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double throughput_gbps;
};

// Structure to hold paired dispatch+combine benchmark results
struct PairedBenchResult {
    BenchResult dispatch;
    BenchResult combine;
    BenchResult total;
};

// Run paired dispatch+combine benchmark with separate timing for each phase
// This ensures dispatch and combine are always paired (required for correctness)
// while still measuring individual performance
PairedBenchResult runPairedBenchmark(
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    int num_warmup,
    int num_iters,
    size_t dispatch_bytes,
    size_t combine_bytes,
    cudaStream_t stream,
    ncclComm_t comm,
    void* barrier_workspace
) {
    // Warmup with paired dispatch+combine
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    for (int i = 0; i < num_warmup; i++) {
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        ncclBarrier(comm, stream, barrier_workspace);
    }

    // Create events for dispatch, combine, and total timing
    std::vector<cudaEvent_t> dispatch_start(num_iters);
    std::vector<cudaEvent_t> dispatch_end(num_iters);
    std::vector<cudaEvent_t> combine_start(num_iters);
    std::vector<cudaEvent_t> combine_end(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventCreate(&dispatch_start[i]));
        CUDACHECK(cudaEventCreate(&dispatch_end[i]));
        CUDACHECK(cudaEventCreate(&combine_start[i]));
        CUDACHECK(cudaEventCreate(&combine_end[i]));
    }

    // Run paired benchmark with individual timing
    // Events are recorded immediately after kernel launch (before sync) to measure GPU time only
    // Sync happens after event recording to not affect timing
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventRecord(dispatch_start[i], stream));
        dispatch_fn();
        CUDACHECK(cudaEventRecord(dispatch_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));              // Sync outside timing
        CUDACHECK(cudaEventRecord(combine_start[i], stream));  // Record after sync, before combine
        combine_fn();
        CUDACHECK(cudaEventRecord(combine_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));             // Sync outside timing
        ncclBarrier(comm, stream, barrier_workspace);
    }

    // Collect times
    std::vector<float> dispatch_times(num_iters);
    std::vector<float> combine_times(num_iters);
    std::vector<float> total_times(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventElapsedTime(&dispatch_times[i], dispatch_start[i], dispatch_end[i]));
        CUDACHECK(cudaEventElapsedTime(&combine_times[i], combine_start[i], combine_end[i]));
        CUDACHECK(cudaEventElapsedTime(&total_times[i], dispatch_start[i], combine_end[i]));
    }

    // Cleanup events
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventDestroy(dispatch_start[i]));
        CUDACHECK(cudaEventDestroy(dispatch_end[i]));
        CUDACHECK(cudaEventDestroy(combine_start[i]));
        CUDACHECK(cudaEventDestroy(combine_end[i]));
    }

    // Helper to calculate stats from times vector (skip first iteration if we have more than 1)
    auto calc_stats = [](const std::vector<float>& times, size_t data_bytes) -> BenchResult {
        // For HT mode with only 1 iteration, don't skip any - use all data
        // For LL mode with multiple iterations, skip the first (warmup outlier)
        std::vector<float> times_trimmed;
        if (times.size() > 1) {
            times_trimmed.assign(times.begin() + 1, times.end());
        } else {
            times_trimmed = times;  // Use all data when we only have 1 iteration
        }

        BenchResult result;
        if (times_trimmed.empty()) {
            result.avg_ms = 0;
            result.min_ms = 0;
            result.max_ms = 0;
            result.throughput_gbps = 0;
        } else {
            result.avg_ms = std::accumulate(times_trimmed.begin(), times_trimmed.end(), 0.0) / times_trimmed.size();
            result.min_ms = *std::min_element(times_trimmed.begin(), times_trimmed.end());
            result.max_ms = *std::max_element(times_trimmed.begin(), times_trimmed.end());
            result.throughput_gbps = (data_bytes / 1e9) / (result.avg_ms / 1000.0);
        }
        return result;
    };

    PairedBenchResult result;
    result.dispatch = calc_stats(dispatch_times, dispatch_bytes);
    result.combine = calc_stats(combine_times, combine_bytes);
    result.total = calc_stats(total_times, dispatch_bytes + combine_bytes);

    return result;
}

// Structure to hold Low Latency byte calculation
// Matches DeepEP test_low_latency.py methodology
struct LowLatencyBytes {
    size_t dispatch_bytes;  // FP8 or BF16 format per selection
    size_t combine_bytes;   // BF16 format: hidden * 2 per selection
    unsigned int num_valid_selections;
    bool is_fp8;  // Whether dispatch uses FP8
};

// Calculate bytes for Low Latency mode
// Dispatch can be FP8 or BF16, combine is always BF16
LowLatencyBytes calculateLowLatencyBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int hidden,
    bool use_fp8
) {
    LowLatencyBytes bytes = {0, 0, 0, use_fp8};

    // Count valid selections (non-masked entries)
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        if (topk_idx_host[i] >= 0) {
            bytes.num_valid_selections++;
        }
    }

    // FP8 bytes per selection: hidden + hidden/128*4 + 16 (scale factors + metadata)
    const size_t fp8_bytes_per_selection = hidden + (hidden / 128) * 4 + 16;
    // BF16 bytes per selection: hidden * 2
    const size_t bf16_bytes_per_selection = hidden * 2;

    // Dispatch: FP8 or BF16 based on config
    bytes.dispatch_bytes = static_cast<size_t>(bytes.num_valid_selections) *
                           (use_fp8 ? fp8_bytes_per_selection : bf16_bytes_per_selection);
    // Combine: always BF16
    bytes.combine_bytes = static_cast<size_t>(bytes.num_valid_selections) * bf16_bytes_per_selection;

    return bytes;
}

// Six bandwidth metrics for High Throughput mode, all dividing by measured time t:
//
//  Send-side (this rank dispatching tokens to experts):
//   total_send  = total_send_bytes / t   — all destinations (NVL+RDMA)
//   nvl_send    = nvl_send_bytes / t     — local node only (NVLink)
//   rdma_send   = rdma_send_bytes / t    — remote nodes only (RDMA outbound)
//
//  Recv-side (this rank's experts receiving tokens):
//   total_recv  = total_recv_bytes / t   — all sources (NVL+RDMA)
//   nvl_recv    = nvl_recv_bytes / t     — from local ranks (NVLink)
//   rdma_recv   = rdma_recv_bytes / t    — from remote ranks (RDMA inbound)
//
//  Derived: nvl_send = total_send - rdma_send
//           nvl_recv = total_recv - rdma_recv
struct HighThroughputBytes {
    size_t total_send_bytes;     // NVL + RDMA outbound
    size_t rdma_send_bytes;      // RDMA outbound only
    size_t total_recv_bytes;     // NVL + RDMA inbound
    size_t rdma_recv_bytes;      // RDMA inbound only (from remote ranks)
    unsigned int total_send_tokens;
    unsigned int rdma_send_tokens;
    unsigned int rdma_recv_tokens;
    unsigned int total_recv_tokens;
    bool is_fp8;
};

// Calculate all six byte metrics from topk_idx for High Throughput mode.
//
// Send side: count unique (token, node) pairs this rank sends to.
//   total_send_tokens = all nodes (local + remote)
//   rdma_send_tokens  = remote nodes only
//
// Recv side: simulate all source ranks' randperm routing (deterministic from
// seed = src_rank + 42) to count unique (src_rank, token) pairs where at least
// one selected expert belongs to myRank.
//   total_recv_tokens = all source ranks (NVL + RDMA)
//   rdma_recv_tokens = remote source ranks only
HighThroughputBytes calculateHighThroughputBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int hidden,
    int myRank,
    int nRanks,
    bool use_fp8,
    int num_ranks_per_node
) {
    HighThroughputBytes bytes = {0, 0, 0, 0, 0, 0, 0, 0, use_fp8};

    int num_nodes = (nRanks + num_ranks_per_node - 1) / num_ranks_per_node;
    unsigned int num_experts_per_node = static_cast<unsigned int>(num_experts / num_nodes);
    int local_node = myRank / num_ranks_per_node;
    unsigned int num_experts_per_rank = num_experts / static_cast<unsigned int>(nRanks);

    // Send side: count unique (token, node) pairs from this rank's topk_idx
    // A token routed to multiple experts on the same node is counted only once, even though
    // NCCL EP sends it to each target rank individually via NVLink P2P (not once per node).
    // TODO: switch to per-rank counting for both nvl_send and nvl_recv.
    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> nodes_for_token;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id < 0) continue;
            int target_node = static_cast<int>(expert_id / num_experts_per_node);
            if (nodes_for_token.insert(target_node).second) {
                bytes.total_send_tokens++;
                if (target_node != local_node)
                    bytes.rdma_send_tokens++;
            }
        }
    }

    // Recv side: replay every source rank's randperm routing to count tokens
    // received by myRank. This is deterministic because each rank uses the
    // same seed (src_rank + 42) and same shuffle algorithm.
    // Each (src_rank, token) pair is counted once regardless of how many experts on myRank it targets.
    std::vector<int64_t> src_perm(num_experts);
    for (int src_rank = 0; src_rank < nRanks; src_rank++) {
        int src_node = src_rank / num_ranks_per_node;
        bool is_rdma = (src_node != local_node);

        std::mt19937 src_gen(src_rank + 42);
        std::iota(src_perm.begin(), src_perm.end(), 0);
        for (unsigned int t = 0; t < num_tokens; t++) {
            std::shuffle(src_perm.begin(), src_perm.end(), src_gen);
            for (unsigned int k = 0; k < top_k; k++) {
                int target_rank = static_cast<int>(src_perm[k] / num_experts_per_rank);
                if (target_rank == myRank) {
                    bytes.total_recv_tokens++;
                    if (is_rdma) bytes.rdma_recv_tokens++;
                    break;
                }
            }
        }
    }

    const size_t bf16_bytes_per_token = hidden * 2;
    const double fp8_factor = (1.0 + 4.0 / 128.0) / 2.0;
    const size_t bytes_per_token = use_fp8 ?
        static_cast<size_t>(bf16_bytes_per_token * fp8_factor) : bf16_bytes_per_token;

    bytes.total_send_bytes = bytes.total_send_tokens * bytes_per_token;
    bytes.rdma_send_bytes  = bytes.rdma_send_tokens  * bytes_per_token;
    bytes.total_recv_bytes = bytes.total_recv_tokens   * bytes_per_token;
    bytes.rdma_recv_bytes  = bytes.rdma_recv_tokens   * bytes_per_token;

    return bytes;
}

// Run NVTX profiling with labeled ranges for nsys analysis.
// Profiles one HandleCreate (to see AG + metadata processing) followed by
// num_iters paired Dispatch+Combine iterations.
void runNvtxProfiling(
    int myRank,
    int num_iters,
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    std::function<void()> handle_create_fn,
    cudaStream_t stream,
    ncclComm_t comm,
    void* barrier_workspace
) {
    if (myRank == 0) {
        printf("\n=== NVTX Profiling Mode ===\n");
        printf("Run with: nsys profile --stats=true torchrun ...\n\n");
    }

    ncclBarrier(comm, stream, barrier_workspace);
    CUDACHECK(cudaStreamSynchronize(stream));

    // Start CUDA profiler (for nsys --capture-range=cudaProfilerApi)
    cudaProfilerStart();

    // Profile HandleCreate to expose AG and metadata processing phases
    nvtxRangePush("HandleCreate");
    handle_create_fn();
    CUDACHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    ncclBarrier(comm, stream, barrier_workspace);

    // Profile paired dispatch+combine iterations with individual labels
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    nvtxRangePush("Paired Dispatch+Combine Benchmark");
    for (int i = 0; i < num_iters; i++) {
        nvtxRangePush("Dispatch");
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        nvtxRangePush("Combine");
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        ncclBarrier(comm, stream, barrier_workspace);
    }
    nvtxRangePop();  // Paired Dispatch+Combine Benchmark

    cudaProfilerStop();

    if (myRank == 0) {
        printf("Profiling complete. Analyze with nsys-ui or nsys stats.\n");
    }
}

// Generate topk indices for LL mode (consistent with DeepEP test_low_latency.py)
// abs(randn)+1 scores → topk selection → random -1 masking (simulates dropped tokens)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank,
    int seed
) {
    // Seed with (seed + rank) for reproducibility across ranks
    std::mt19937 gen(seed + rank);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::pair<float, int>> score_idx(num_experts);

    for (unsigned int i = 0; i < num_tokens; i++) {
        // Generate random scores: abs(randn) + 1
        for (unsigned int e = 0; e < num_experts; e++) {
            float score = std::abs(dist(gen)) + 1.0f;
            score_idx[e] = {score, static_cast<int>(e)};
        }

        // Partial sort to get top-k (largest scores first)
        std::partial_sort(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Extract top-k expert indices (sorted by score, descending)
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = score_idx[j].second;
        }
    }

    // Randomly mask 10 positions with -1 (simulates dropped tokens)
    std::uniform_int_distribution<unsigned int> token_dist(0, num_tokens - 1);
    std::uniform_int_distribution<unsigned int> topk_dist(0, top_k - 1);
    for (int i = 0; i < 10; i++) {
        unsigned int token_idx = token_dist(gen);
        unsigned int k_idx = topk_dist(gen);
        topk_idx_host[token_idx * top_k + k_idx] = -1;
    }
}

static_assert(sizeof(ncclUniqueId) == EP_BENCH_UNIQUE_ID_BYTES, "Unexpected ncclUniqueId size");

static void copyBenchResult(const BenchResult& src, EpBenchBenchResult* dst) {
    dst->avg_ms = src.avg_ms;
    dst->min_ms = src.min_ms;
    dst->max_ms = src.max_ms;
    dst->throughput_gbps = src.throughput_gbps;
}

static void runEpBenchImpl(
    const EpBenchBootstrap* bootstrap,
    const EpBenchConfig* bench_config,
    EpBenchLocalResult* result) {
    if (bootstrap == nullptr || bench_config == nullptr || result == nullptr) {
        throw std::runtime_error("ep_bench_run received a null pointer argument");
    }

    const int myRank = bootstrap->rank;
    const int nRanks = bootstrap->world_size;
    const int localRank = bootstrap->local_rank;
    if (nRanks <= 0) {
        throw std::runtime_error("world_size must be positive");
    }
    if (myRank < 0 || myRank >= nRanks) {
        throw std::runtime_error("rank is out of range");
    }
    if (localRank < 0) {
        throw std::runtime_error("local_rank must be non-negative");
    }
    if (bench_config->num_warmup < 0 || bench_config->num_iters < 0) {
        throw std::runtime_error("warmup/iters must be non-negative");
    }

    ncclEpAlgorithm_t algorithm;
    switch (bench_config->algorithm) {
        case NCCL_EP_ALGO_LOW_LATENCY:
        case NCCL_EP_ALGO_HIGH_THROUGHPUT:
            algorithm = static_cast<ncclEpAlgorithm_t>(bench_config->algorithm);
            break;
        default:
            throw std::runtime_error("invalid algorithm value");
    }

    unsigned int num_tokens = bench_config->num_tokens;
    const unsigned int hidden = bench_config->hidden;
    const unsigned int top_k = bench_config->top_k;
    const unsigned int num_experts = bench_config->num_experts;
    const int num_warmup = bench_config->num_warmup;
    const int num_iters = bench_config->num_iters;
    const bool profile_mode = bench_config->profile_mode != 0;
    const bool disable_nvlink = bench_config->disable_nvlink != 0;
    const bool use_fp8 = bench_config->use_fp8 != 0;
    const bool validate_data = bench_config->validate_data != 0;
    const bool dynamic_tokens = bench_config->dynamic_tokens != 0;

    if (num_tokens == 0) {
        num_tokens = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 4096 : 128;
    }
    if (num_experts == 0 || hidden == 0 || top_k == 0) {
        throw std::runtime_error("tokens/hidden/top-k/experts must be positive");
    }
    if (num_experts % static_cast<unsigned int>(nRanks) != 0) {
        throw std::runtime_error("num_experts must be divisible by world_size");
    }
    if (dynamic_tokens) {
        if (algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT) {
            throw std::runtime_error("--dynamic-tokens is only applicable to HT mode (--algorithm ht)");
        }
        throw std::runtime_error(
            "--dynamic-tokens (NCCL_EP_AUTO for max_tokens_per_rank) is not yet supported. "
            "This feature will be available in a future release for HT mode.");
    }

    const unsigned int num_local_experts = num_experts / static_cast<unsigned int>(nRanks);

    if (disable_nvlink && algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "1", 1);
    }

    CUDACHECK(cudaSetDevice(localRank));
    cudaStream_t stream = nullptr;
    CUDACHECK(cudaStreamCreate(&stream));

    ncclUniqueId id = {};
    memcpy(id.internal, bootstrap->nccl_unique_id, EP_BENCH_UNIQUE_ID_BYTES);

    ncclComm_t comm = nullptr;
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    void* barrier_workspace = nullptr;
    CUDACHECK(cudaMalloc(&barrier_workspace, sizeof(int)));

    ncclEpGroup_t ep_group = nullptr;
    ncclEpGroupConfig_t config = {};
    config.version = 1;
    config.algorithm = algorithm;
    config.num_experts = num_experts;
    config.max_tokens_per_rank = dynamic_tokens ? NCCL_EP_AUTO : num_tokens;
    config.token_size_bytes = hidden * 2;
    config.rdma_buffer_size = NCCL_EP_AUTO;
    config.num_qp_per_rank = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? num_local_experts : NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;

    ncclBarrier(comm, stream, barrier_workspace);
    auto group_create_start = std::chrono::steady_clock::now();
    NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config, stream, cudaAllocCallback, cudaFreeCallback));
    CUDACHECK(cudaStreamSynchronize(stream));
    auto group_create_end = std::chrono::steady_clock::now();
    double group_create_ms = elapsedMs(group_create_start, group_create_end);

    ncclNDTensor_t topk_idx = nullptr;
    {
        void* topk_idx_data = nullptr;
        CUDACHECK(cudaMalloc(&topk_idx_data, num_tokens * top_k * sizeof(int64_t)));
        NCCLCHECK(ncclEpTensorCreate(
            ep_group,
            &topk_idx,
            2,
            ncclInt64,
            NCCL_EP_TENSOR_TAG_TOPK_IDX,
            topk_idx_data,
            num_tokens,
            top_k));
    }

    int64_t* topk_idx_host = new int64_t[num_tokens * top_k];
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        generateTopkIndicesHT(topk_idx_host, num_tokens, num_experts, top_k, myRank);
    } else {
        generateRandomTopkIndicesLL(topk_idx_host, num_tokens, num_experts, top_k, myRank);
    }

    LowLatencyBytes ll_bytes = {};
    HighThroughputBytes ht_bytes = {};
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ll_bytes = calculateLowLatencyBytes(topk_idx_host, num_tokens, top_k, hidden, use_fp8);
    } else {
        ht_bytes = calculateHighThroughputBytes(
            topk_idx_host,
            num_tokens,
            top_k,
            num_experts,
            hidden,
            myRank,
            nRanks,
            use_fp8,
            ncclTeamLsa(comm).nRanks);
    }

    {
        void* topk_idx_data = nullptr;
        NCCLCHECK(ncclEpTensorGetData(topk_idx, &topk_idx_data));
        CUDACHECK(cudaMemcpy(
            topk_idx_data,
            topk_idx_host,
            num_tokens * top_k * sizeof(int64_t),
            cudaMemcpyHostToDevice));
    }

    ncclNDTensor_t recv_expert_counter_tensor = nullptr;
    if (dynamic_tokens) {
        void* recv_expert_counter_data = nullptr;
        CUDACHECK(cudaHostAlloc(
            &recv_expert_counter_data,
            num_local_experts * sizeof(int),
            cudaHostAllocMapped));
        NCCLCHECK(ncclEpTensorCreate(
            ep_group,
            &recv_expert_counter_tensor,
            1,
            ncclInt32,
            NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST,
            recv_expert_counter_data,
            num_local_experts));
    }

    ncclEpHandle_t ep_handle = nullptr;
    ncclNDTensor_t handle_local_tensors[1] = {recv_expert_counter_tensor};
    unsigned int handle_num_local_tensors = recv_expert_counter_tensor ? 1u : 0u;
    ncclBarrier(comm, stream, barrier_workspace);
    auto handle_create_start = std::chrono::steady_clock::now();
    NCCLCHECK(ncclEpCreateHandle(
        &ep_handle,
        ep_group,
        topk_idx,
        handle_local_tensors,
        handle_num_local_tensors,
        nullptr,
        stream,
        use_fp8));
    CUDACHECK(cudaStreamSynchronize(stream));
    auto handle_create_end = std::chrono::steady_clock::now();
    double handle_create_ms = elapsedMs(handle_create_start, handle_create_end);

    unsigned int num_recv_tokens = 0;
    if (dynamic_tokens) {
        NCCLCHECK(ncclEpHandleGetNumRecvTokens(ep_handle, &num_recv_tokens));
    } else {
        num_recv_tokens = config.max_tokens_per_rank * num_local_experts;
    }
    if (num_recv_tokens == 0) {
        throw std::runtime_error("num_recv_tokens resolved to zero");
    }

    BenchmarkTensors tensors = {};
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        setupLowLatencyTensors(
            ep_group,
            tensors,
            topk_idx,
            num_tokens,
            hidden,
            top_k,
            num_local_experts,
            config.max_tokens_per_rank,
            nRanks);
    } else {
        setupHighThroughputTensors(
            ep_group,
            tensors,
            topk_idx,
            num_tokens,
            hidden,
            top_k,
            num_local_experts,
            num_recv_tokens);
    }

    if (validate_data) {
        initializeValidationData(
            tensors,
            num_tokens,
            hidden,
            top_k,
            myRank,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
    }

    ncclEpDispatchConfig_t dispatch_config = {};
    dispatch_config.round_scales = 0;

    ncclBarrier(comm, stream, barrier_workspace);
    CUDACHECK(cudaStreamSynchronize(stream));

    size_t dispatch_data_bytes = 0;
    size_t combine_data_bytes = 0;
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        dispatch_data_bytes = ll_bytes.dispatch_bytes;
        combine_data_bytes = ll_bytes.combine_bytes;
    } else {
        dispatch_data_bytes = ht_bytes.rdma_send_bytes + ht_bytes.total_recv_bytes;
        combine_data_bytes = dispatch_data_bytes;
    }

    const int num_dispatch_local_tensors = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 0 : 1;
    auto dispatch_fn = [&]() {
        NCCLCHECK(ncclEpDispatch(
            ep_handle,
            tensors.inputs,
            tensors.num_dispatch_inputs,
            tensors.outputs,
            tensors.num_dispatch_outputs,
            tensors.local_tensors,
            num_dispatch_local_tensors,
            false,
            &dispatch_config,
            stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    auto combine_fn = [&]() {
        NCCLCHECK(ncclEpCombine(
            ep_handle,
            tensors.combine_inputs,
            tensors.num_combine_inputs,
            tensors.combine_outputs,
            tensors.num_combine_outputs,
            tensors.combine_local_tensors,
            tensors.num_combine_local_tensors,
            false,
            nullptr,
            stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    KernelTimer ktimer;
    ktimer.start();
    ncclBarrier(comm, stream, barrier_workspace);
    PairedBenchResult paired_result = runPairedBenchmark(
        dispatch_fn,
        combine_fn,
        num_warmup,
        num_iters,
        dispatch_data_bytes,
        combine_data_bytes,
        stream,
        comm,
        barrier_workspace);
    ktimer.stop();

    if (profile_mode) {
        auto handle_create_fn = [&]() {
            NCCLCHECK(ncclEpHandleDestroy(ep_handle));
            NCCLCHECK(ncclEpCreateHandle(
                &ep_handle,
                ep_group,
                topk_idx,
                handle_local_tensors,
                handle_num_local_tensors,
                nullptr,
                stream,
                use_fp8));
        };
        runNvtxProfiling(
            myRank,
            num_iters,
            dispatch_fn,
            combine_fn,
            handle_create_fn,
            stream,
            comm,
            barrier_workspace);
    }

    ValidationResult dispatch_valid = {true, 0, 0.0, ""};
    ValidationResult combine_valid = {true, 0, 0.0, ""};
    if (validate_data) {
        ncclBarrier(comm, stream, barrier_workspace);
        initializeValidationData(
            tensors,
            num_tokens,
            hidden,
            top_k,
            myRank,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);

        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        ncclBarrier(comm, stream, barrier_workspace);

        dispatch_valid = validateDispatchOutput(
            tensors,
            num_tokens,
            hidden,
            top_k,
            num_experts,
            num_local_experts,
            myRank,
            nRanks,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);

        {
            void* eo_data = nullptr;
            void* output0_data = nullptr;
            NCCLCHECK(ncclEpTensorGetData(tensors.expert_outputs, &eo_data));
            NCCLCHECK(ncclEpTensorGetData(tensors.outputs[0], &output0_data));

            if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
                const unsigned int* eo_sizes;
                unsigned int eo_ndim;
                NCCLCHECK(ncclEpTensorGetSizes(tensors.expert_outputs, &eo_sizes, &eo_ndim));
                size_t data_size = eo_sizes[0] * eo_sizes[1] * sizeof(uint16_t);
                CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
            } else {
                const unsigned int* out0_sizes;
                unsigned int out0_ndim;
                NCCLCHECK(ncclEpTensorGetSizes(tensors.outputs[0], &out0_sizes, &out0_ndim));
                size_t data_size = out0_sizes[0] * out0_sizes[1] * out0_sizes[2] * sizeof(uint16_t);
                CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
            }
        }

        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        ncclBarrier(comm, stream, barrier_workspace);

        combine_valid = validateCombineOutput(
            tensors,
            num_tokens,
            hidden,
            top_k,
            num_experts,
            myRank,
            nRanks,
            algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT,
            topk_idx_host);
    }

    result->algorithm = algorithm;
    copyBenchResult(paired_result.dispatch, &result->dispatch);
    copyBenchResult(paired_result.combine, &result->combine);
    copyBenchResult(paired_result.total, &result->total);
    result->group_create_ms = group_create_ms;
    result->handle_create_ms = handle_create_ms;
    result->dispatch_kernel_us = ktimer.get_avg_us("dispatch_kernel");
    result->combine_kernel_us = ktimer.get_avg_us("combine_kernel");
    result->ll_dispatch_bytes = static_cast<uint64_t>(ll_bytes.dispatch_bytes);
    result->ll_combine_bytes = static_cast<uint64_t>(ll_bytes.combine_bytes);
    result->ll_num_valid_selections = ll_bytes.num_valid_selections;
    result->ll_is_fp8 = ll_bytes.is_fp8 ? 1 : 0;
    result->ht_total_send_bytes = static_cast<uint64_t>(ht_bytes.total_send_bytes);
    result->ht_rdma_send_bytes = static_cast<uint64_t>(ht_bytes.rdma_send_bytes);
    result->ht_total_recv_bytes = static_cast<uint64_t>(ht_bytes.total_recv_bytes);
    result->ht_rdma_recv_bytes = static_cast<uint64_t>(ht_bytes.rdma_recv_bytes);
    result->ht_total_send_tokens = ht_bytes.total_send_tokens;
    result->ht_rdma_send_tokens = ht_bytes.rdma_send_tokens;
    result->ht_total_recv_tokens = ht_bytes.total_recv_tokens;
    result->ht_rdma_recv_tokens = ht_bytes.rdma_recv_tokens;
    result->ht_is_fp8 = ht_bytes.is_fp8 ? 1 : 0;
    result->dispatch_validation_pass = dispatch_valid.passed ? 1 : 0;
    result->combine_validation_pass = combine_valid.passed ? 1 : 0;
    result->combine_validation_max_diff = combine_valid.max_diff;

    cleanupBenchmarkTensors(ep_group, tensors, topk_idx);
    delete[] topk_idx_host;

    NCCLCHECK(ncclEpHandleDestroy(ep_handle));
    if (dynamic_tokens && recv_expert_counter_tensor != nullptr) {
        void* rec_data = nullptr;
        NCCLCHECK(ncclEpTensorGetData(recv_expert_counter_tensor, &rec_data));
        if (rec_data) {
            CUDACHECK(cudaFreeHost(rec_data));
        }
        NCCLCHECK(ncclEpTensorDestroy(ep_group, recv_expert_counter_tensor));
    }

    NCCLCHECK(ncclEpGroupDestroy(ep_group, stream));
    CUDACHECK(cudaFree(barrier_workspace));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(stream));
}

extern "C" int ep_bench_run(
    const EpBenchBootstrap* bootstrap,
    const EpBenchConfig* config,
    EpBenchLocalResult* result) {
    if (result != nullptr) {
        memset(result, 0, sizeof(*result));
    }

    try {
        runEpBenchImpl(bootstrap, config, result);
        return 0;
    } catch (const std::exception& e) {
        if (result != nullptr) {
            result->error_code = 1;
            snprintf(result->error_message, sizeof(result->error_message), "%s", e.what());
        }
        return 1;
    } catch (...) {
        if (result != nullptr) {
            result->error_code = 1;
            snprintf(result->error_message, sizeof(result->error_message), "unknown ep_bench_run failure");
        }
        return 1;
    }
}
