#ifndef NCCL_EP_EP_BENCH_PYTHON_CORE_H_
#define NCCL_EP_EP_BENCH_PYTHON_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EP_BENCH_UNIQUE_ID_BYTES 128
#define EP_BENCH_ERROR_MESSAGE_BYTES 1024

typedef struct EpBenchBootstrap {
  int rank;
  int world_size;
  int local_rank;
  unsigned char nccl_unique_id[EP_BENCH_UNIQUE_ID_BYTES];
} EpBenchBootstrap;

typedef struct EpBenchConfig {
  int algorithm;
  unsigned int num_tokens;
  unsigned int hidden;
  unsigned int top_k;
  unsigned int num_experts;
  int num_warmup;
  int num_iters;
  int profile_mode;
  int disable_nvlink;
  int use_fp8;
  int validate_data;
  int dynamic_tokens;
} EpBenchConfig;

typedef struct EpBenchBenchResult {
  double avg_ms;
  double min_ms;
  double max_ms;
  double throughput_gbps;
} EpBenchBenchResult;

typedef struct EpBenchLocalResult {
  int algorithm;
  EpBenchBenchResult dispatch;
  EpBenchBenchResult combine;
  EpBenchBenchResult total;
  double group_create_ms;
  double handle_create_ms;
  double dispatch_kernel_us;
  double combine_kernel_us;
  uint64_t ll_dispatch_bytes;
  uint64_t ll_combine_bytes;
  unsigned int ll_num_valid_selections;
  int ll_is_fp8;
  uint64_t ht_total_send_bytes;
  uint64_t ht_rdma_send_bytes;
  uint64_t ht_total_recv_bytes;
  uint64_t ht_rdma_recv_bytes;
  unsigned int ht_total_send_tokens;
  unsigned int ht_rdma_send_tokens;
  unsigned int ht_total_recv_tokens;
  unsigned int ht_rdma_recv_tokens;
  int ht_is_fp8;
  int dispatch_validation_pass;
  int combine_validation_pass;
  double combine_validation_max_diff;
  int error_code;
  char error_message[EP_BENCH_ERROR_MESSAGE_BYTES];
} EpBenchLocalResult;

int ep_bench_run(
    const EpBenchBootstrap* bootstrap,
    const EpBenchConfig* config,
    EpBenchLocalResult* result);

#ifdef __cplusplus
}
#endif

#endif
