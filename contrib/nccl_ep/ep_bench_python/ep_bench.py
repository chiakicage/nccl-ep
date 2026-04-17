#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torchrun launcher for the MPI-free NCCL EP benchmark core."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
PYTHON_BINDINGS_DIR = THIS_DIR.parent / "python"

try:
    from nccl_ep import NCCLLibrary, ncclEpAlgorithm_t
except ImportError:
    sys.path.insert(0, str(PYTHON_BINDINGS_DIR))
    from nccl_ep import NCCLLibrary, ncclEpAlgorithm_t


EP_BENCH_UNIQUE_ID_BYTES = 128
EP_BENCH_ERROR_MESSAGE_BYTES = 1024


class EpBenchBootstrap(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_int),
        ("world_size", ctypes.c_int),
        ("local_rank", ctypes.c_int),
        ("nccl_unique_id", ctypes.c_ubyte * EP_BENCH_UNIQUE_ID_BYTES),
    ]


class EpBenchConfig(ctypes.Structure):
    _fields_ = [
        ("algorithm", ctypes.c_int),
        ("num_tokens", ctypes.c_uint),
        ("hidden", ctypes.c_uint),
        ("top_k", ctypes.c_uint),
        ("num_experts", ctypes.c_uint),
        ("num_warmup", ctypes.c_int),
        ("num_iters", ctypes.c_int),
        ("profile_mode", ctypes.c_int),
        ("disable_nvlink", ctypes.c_int),
        ("use_fp8", ctypes.c_int),
        ("validate_data", ctypes.c_int),
        ("dynamic_tokens", ctypes.c_int),
    ]


class EpBenchBenchResult(ctypes.Structure):
    _fields_ = [
        ("avg_ms", ctypes.c_double),
        ("min_ms", ctypes.c_double),
        ("max_ms", ctypes.c_double),
        ("throughput_gbps", ctypes.c_double),
    ]


class EpBenchLocalResult(ctypes.Structure):
    _fields_ = [
        ("algorithm", ctypes.c_int),
        ("dispatch", EpBenchBenchResult),
        ("combine", EpBenchBenchResult),
        ("total", EpBenchBenchResult),
        ("group_create_ms", ctypes.c_double),
        ("handle_create_ms", ctypes.c_double),
        ("dispatch_kernel_us", ctypes.c_double),
        ("combine_kernel_us", ctypes.c_double),
        ("ll_dispatch_bytes", ctypes.c_uint64),
        ("ll_combine_bytes", ctypes.c_uint64),
        ("ll_num_valid_selections", ctypes.c_uint),
        ("ll_is_fp8", ctypes.c_int),
        ("ht_total_send_bytes", ctypes.c_uint64),
        ("ht_rdma_send_bytes", ctypes.c_uint64),
        ("ht_total_recv_bytes", ctypes.c_uint64),
        ("ht_rdma_recv_bytes", ctypes.c_uint64),
        ("ht_total_send_tokens", ctypes.c_uint),
        ("ht_rdma_send_tokens", ctypes.c_uint),
        ("ht_total_recv_tokens", ctypes.c_uint),
        ("ht_rdma_recv_tokens", ctypes.c_uint),
        ("ht_is_fp8", ctypes.c_int),
        ("dispatch_validation_pass", ctypes.c_int),
        ("combine_validation_pass", ctypes.c_int),
        ("combine_validation_max_diff", ctypes.c_double),
        ("error_code", ctypes.c_int),
        ("error_message", ctypes.c_char * EP_BENCH_ERROR_MESSAGE_BYTES),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Torchrun launcher for NCCL EP benchmark")
    parser.add_argument(
        "--algorithm",
        default="ll",
        choices=("ll", "low-latency", "ht", "high-throughput"),
        help="Algorithm mode (default: ll)",
    )
    parser.add_argument("--tokens", type=int, default=0, help="Number of tokens (default: LL=128, HT=4096)")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension (default: 7168)")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k experts per token (default: 8)")
    parser.add_argument("--experts", type=int, default=256, help="Total experts (default: 256)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations (default: 50)")
    parser.add_argument("--profile", action="store_true", help="Enable NVTX profiling mode")
    parser.add_argument(
        "--disable-nvlink",
        action="store_true",
        help="Disable NVLink and force RDMA for intranode LL communication",
    )
    parser.add_argument("--use-fp8", action="store_true", help="Use FP8 for dispatch")
    parser.add_argument("--validate", action="store_true", help="Enable dispatch/combine validation")
    parser.add_argument(
        "--dynamic-tokens",
        action="store_true",
        help="Enable dynamic token allocation (currently unsupported, kept for parity)",
    )
    return parser


def normalize_algorithm(value: str) -> int:
    if value in {"ll", "low-latency"}:
        return ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
    return ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT


def effective_num_tokens(algorithm: int, requested: int) -> int:
    if requested > 0:
        return requested
    return 4096 if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT else 128


def require_torchrun_env() -> tuple[int, int, int]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing torchrun environment variables: {missing_str}")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def load_core_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(THIS_DIR / "libep_bench_core.so"))
    lib.ep_bench_run.argtypes = [
        ctypes.POINTER(EpBenchBootstrap),
        ctypes.POINTER(EpBenchConfig),
        ctypes.POINTER(EpBenchLocalResult),
    ]
    lib.ep_bench_run.restype = ctypes.c_int
    return lib


def get_unique_id_bytes(rank: int) -> list[int]:
    if rank == 0:
        nccl_lib = NCCLLibrary()
        unique_id = nccl_lib.ncclGetUniqueId()
        raw = ctypes.string_at(ctypes.byref(unique_id), EP_BENCH_UNIQUE_ID_BYTES)
        tensor = torch.tensor(list(raw), dtype=torch.uint8)
    else:
        tensor = torch.zeros(EP_BENCH_UNIQUE_ID_BYTES, dtype=torch.uint8)
    dist.broadcast(tensor, src=0)
    return tensor.tolist()


def to_plain_bench(result: EpBenchBenchResult) -> dict[str, float]:
    return {
        "avg_ms": float(result.avg_ms),
        "min_ms": float(result.min_ms),
        "max_ms": float(result.max_ms),
        "throughput_gbps": float(result.throughput_gbps),
    }


def decode_error_message(result: EpBenchLocalResult) -> str:
    return bytes(result.error_message).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def local_result_to_dict(rank: int, result: EpBenchLocalResult, rc: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "rc": int(rc),
        "algorithm": int(result.algorithm),
        "dispatch": to_plain_bench(result.dispatch),
        "combine": to_plain_bench(result.combine),
        "total": to_plain_bench(result.total),
        "group_create_ms": float(result.group_create_ms),
        "handle_create_ms": float(result.handle_create_ms),
        "dispatch_kernel_us": float(result.dispatch_kernel_us),
        "combine_kernel_us": float(result.combine_kernel_us),
        "ll_dispatch_bytes": int(result.ll_dispatch_bytes),
        "ll_combine_bytes": int(result.ll_combine_bytes),
        "ll_num_valid_selections": int(result.ll_num_valid_selections),
        "ll_is_fp8": bool(result.ll_is_fp8),
        "ht_total_send_bytes": int(result.ht_total_send_bytes),
        "ht_rdma_send_bytes": int(result.ht_rdma_send_bytes),
        "ht_total_recv_bytes": int(result.ht_total_recv_bytes),
        "ht_rdma_recv_bytes": int(result.ht_rdma_recv_bytes),
        "ht_total_send_tokens": int(result.ht_total_send_tokens),
        "ht_rdma_send_tokens": int(result.ht_rdma_send_tokens),
        "ht_total_recv_tokens": int(result.ht_total_recv_tokens),
        "ht_rdma_recv_tokens": int(result.ht_rdma_recv_tokens),
        "ht_is_fp8": bool(result.ht_is_fp8),
        "dispatch_validation_pass": bool(result.dispatch_validation_pass),
        "combine_validation_pass": bool(result.combine_validation_pass),
        "combine_validation_max_diff": float(result.combine_validation_max_diff),
        "error_code": int(result.error_code),
        "error_message": decode_error_message(result),
    }


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_bw(bytes_count: float, avg_ms: float) -> float:
    if avg_ms <= 0.0:
        return 0.0
    return (bytes_count / 1e9) / (avg_ms / 1000.0)


def print_configuration(args: argparse.Namespace, algorithm: int, world_size: int) -> None:
    effective_tokens = effective_num_tokens(algorithm, args.tokens)
    local_experts = args.experts // world_size if world_size else 0
    algo_name = "LOW_LATENCY" if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY else "HIGH_THROUGHPUT"
    print("=== NCCL EP Performance Benchmark ===")
    print("Configuration:")
    print(f"  Algorithm:       {algo_name}")
    print(f"  Ranks:           {world_size}")
    print(f"  Tokens:          {effective_tokens}")
    print(f"  Hidden:          {args.hidden}")
    print(f"  Top-k:           {args.top_k}")
    print(f"  Experts:         {args.experts} (local: {local_experts})")
    print(f"  Warmup iters:    {args.warmup}")
    print(f"  Benchmark iters: {args.iters}")
    print(f"  Dispatch dtype:  {'FP8' if args.use_fp8 else 'BF16'}")
    print(f"  Profile mode:    {'enabled' if args.profile else 'disabled'}")
    print(
        "  NVLink:          "
        + ("disabled (force RDMA intranode, LL only)" if args.disable_nvlink else "enabled")
    )
    print(f"  Validate mode:   {'enabled' if args.validate else 'disabled'}")
    print(
        "  Dynamic tokens:  "
        + ("enabled (NCCL_EP_AUTO)" if args.dynamic_tokens else "disabled")
    )
    print()


def print_local_lines(results: list[dict[str, Any]], algorithm: int) -> None:
    for item in sorted(results, key=lambda entry: entry["rank"]):
        rank = item["rank"]
        if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY:
            dispatch = item["dispatch"]
            combine = item["combine"]
            total = item["total"]
            print(
                f"[Rank {rank}] Dispatch:         avg={dispatch['avg_ms'] * 1000:.2f} us, "
                f"min={dispatch['min_ms'] * 1000:.2f} us, max={dispatch['max_ms'] * 1000:.2f} us, "
                f"throughput={dispatch['throughput_gbps']:.2f} GB/s"
            )
            print(
                f"[Rank {rank}] Combine:          avg={combine['avg_ms'] * 1000:.2f} us, "
                f"min={combine['min_ms'] * 1000:.2f} us, max={combine['max_ms'] * 1000:.2f} us, "
                f"throughput={combine['throughput_gbps']:.2f} GB/s"
            )
            print(
                f"[Rank {rank}] Dispatch+Combine: avg={total['avg_ms'] * 1000:.2f} us, "
                f"min={total['min_ms'] * 1000:.2f} us, max={total['max_ms'] * 1000:.2f} us, "
                f"throughput={total['throughput_gbps']:.2f} GB/s"
            )
        else:
            print(
                f"[Rank {rank}] Dispatch:         total={item['dispatch']['avg_ms'] * 1000:.2f} us  "
                f"kernel={item['dispatch_kernel_us']:.2f} us"
            )
            print(
                f"[Rank {rank}] Combine:          total={item['combine']['avg_ms'] * 1000:.2f} us  "
                f"kernel={item['combine_kernel_us']:.2f} us"
            )
            print(
                f"[Rank {rank}] Dispatch+Combine: total={item['total']['avg_ms'] * 1000:.2f} us"
            )


def print_ll_summary(results: list[dict[str, Any]]) -> None:
    dispatch_avg = avg([item["dispatch"]["avg_ms"] for item in results])
    dispatch_min = min(item["dispatch"]["min_ms"] for item in results)
    dispatch_max = max(item["dispatch"]["max_ms"] for item in results)
    combine_avg = avg([item["combine"]["avg_ms"] for item in results])
    combine_min = min(item["combine"]["min_ms"] for item in results)
    combine_max = max(item["combine"]["max_ms"] for item in results)
    total_avg = avg([item["total"]["avg_ms"] for item in results])
    total_min = min(item["total"]["min_ms"] for item in results)
    total_max = max(item["total"]["max_ms"] for item in results)

    dispatch_tp_min = min(results, key=lambda item: item["dispatch"]["throughput_gbps"])
    dispatch_tp_max = max(results, key=lambda item: item["dispatch"]["throughput_gbps"])
    combine_tp_min = min(results, key=lambda item: item["combine"]["throughput_gbps"])
    combine_tp_max = max(results, key=lambda item: item["combine"]["throughput_gbps"])
    total_tp_min = min(results, key=lambda item: item["total"]["throughput_gbps"])
    total_tp_max = max(results, key=lambda item: item["total"]["throughput_gbps"])

    ll_dispatch_bytes = results[0]["ll_dispatch_bytes"]
    ll_combine_bytes = results[0]["ll_combine_bytes"]
    ll_is_fp8 = results[0]["ll_is_fp8"]
    selections = results[0]["ll_num_valid_selections"]
    total_data_bytes = ll_dispatch_bytes + ll_combine_bytes

    print(f"\n=== Summary (Low Latency, across {len(results)} ranks) ===")
    print(
        f"Dispatch ({'FP8' if ll_is_fp8 else 'BF16'}):  avg={dispatch_avg * 1000:.2f} us, "
        f"min={dispatch_min * 1000:.2f} us, max={dispatch_max * 1000:.2f} us"
    )
    print(
        "                  throughput: "
        f"avg={safe_bw(ll_dispatch_bytes, dispatch_avg):.2f} GB/s, "
        f"min={dispatch_tp_min['dispatch']['throughput_gbps']:.2f} GB/s (rank {dispatch_tp_min['rank']}), "
        f"max={dispatch_tp_max['dispatch']['throughput_gbps']:.2f} GB/s (rank {dispatch_tp_max['rank']})"
    )
    print(
        f"Combine (BF16):   avg={combine_avg * 1000:.2f} us, "
        f"min={combine_min * 1000:.2f} us, max={combine_max * 1000:.2f} us"
    )
    print(
        "                  throughput: "
        f"avg={safe_bw(ll_combine_bytes, combine_avg):.2f} GB/s, "
        f"min={combine_tp_min['combine']['throughput_gbps']:.2f} GB/s (rank {combine_tp_min['rank']}), "
        f"max={combine_tp_max['combine']['throughput_gbps']:.2f} GB/s (rank {combine_tp_max['rank']})"
    )
    print(
        f"Total (D+C):      avg={total_avg * 1000:.2f} us, "
        f"min={total_min * 1000:.2f} us, max={total_max * 1000:.2f} us"
    )
    print(
        "                  throughput: "
        f"avg={safe_bw(total_data_bytes, total_avg):.2f} GB/s, "
        f"min={total_tp_min['total']['throughput_gbps']:.2f} GB/s (rank {total_tp_min['rank']}), "
        f"max={total_tp_max['total']['throughput_gbps']:.2f} GB/s (rank {total_tp_max['rank']})"
    )
    print(
        f"\nByte counts: dispatch={ll_dispatch_bytes / 1e6:.2f} MB ({'FP8' if ll_is_fp8 else 'BF16'}), "
        f"combine={ll_combine_bytes / 1e6:.2f} MB (BF16), selections={selections}"
    )

    avg_dispatch_kernel = avg([item["dispatch_kernel_us"] for item in results])
    avg_combine_kernel = avg([item["combine_kernel_us"] for item in results])
    print(f"\n=== Kernel-Only Timing (CUPTI, avg across {len(results)} ranks) ===")
    print(f"dispatch_kernel: avg={avg_dispatch_kernel:.2f} us")
    print(f"combine_kernel:  avg={avg_combine_kernel:.2f} us")
    if avg_dispatch_kernel == 0.0 or avg_combine_kernel == 0.0:
        print("  NOTE: 0 us means no matching kernel was captured.")
        print("  Uncomment ktimer.dump() in ep_bench_core.cu to inspect captured kernel names.")


def print_ht_summary(results: list[dict[str, Any]]) -> None:
    dispatch_avg = avg([item["dispatch"]["avg_ms"] for item in results])
    dispatch_min = min(item["dispatch"]["min_ms"] for item in results)
    dispatch_max = max(item["dispatch"]["max_ms"] for item in results)
    combine_avg = avg([item["combine"]["avg_ms"] for item in results])
    combine_min = min(item["combine"]["min_ms"] for item in results)
    combine_max = max(item["combine"]["max_ms"] for item in results)
    total_avg = avg([item["total"]["avg_ms"] for item in results])
    total_min = min(item["total"]["min_ms"] for item in results)
    total_max = max(item["total"]["max_ms"] for item in results)

    avg_total_send = avg([item["ht_total_send_bytes"] for item in results])
    avg_rdma_send = avg([item["ht_rdma_send_bytes"] for item in results])
    avg_total_recv = avg([item["ht_total_recv_bytes"] for item in results])
    avg_rdma_recv = avg([item["ht_rdma_recv_bytes"] for item in results])
    avg_nvl_send = avg_total_send - avg_rdma_send
    avg_nvl_recv = avg_total_recv - avg_rdma_recv

    avg_dispatch_kernel = avg([item["dispatch_kernel_us"] for item in results])
    avg_combine_kernel = avg([item["combine_kernel_us"] for item in results])
    ht_is_fp8 = results[0]["ht_is_fp8"]

    avg_total_send_tokens = round(avg([item["ht_total_send_tokens"] for item in results]))
    avg_rdma_send_tokens = round(avg([item["ht_rdma_send_tokens"] for item in results]))
    avg_total_recv_tokens = round(avg([item["ht_total_recv_tokens"] for item in results]))
    avg_rdma_recv_tokens = round(avg([item["ht_rdma_recv_tokens"] for item in results]))

    dispatch_total_s = dispatch_avg / 1000.0
    combine_total_s = combine_avg / 1000.0
    dispatch_kernel_s = avg_dispatch_kernel / 1e6
    combine_kernel_s = avg_combine_kernel / 1e6

    print(f"\n=== Summary (High Throughput {'FP8' if ht_is_fp8 else 'BF16'}, across {len(results)} ranks) ===")
    print("NOTE: total time = kernel time + memcpyD2D + misc")
    print("--- BW based on total time ---")
    print(
        f"Dispatch:    total={dispatch_avg * 1000:.2f} us "
        f"(min={dispatch_min * 1000:.2f}, max={dispatch_max * 1000:.2f})"
    )
    if dispatch_total_s > 0.0:
        print(
            f"             recv: total_bw={(avg_total_recv / 1e9) / dispatch_total_s:.2f}  "
            f"nvl_bw={(avg_nvl_recv / 1e9) / dispatch_total_s:.2f}  "
            f"rdma_bw={(avg_rdma_recv / 1e9) / dispatch_total_s:.2f} GB/s"
        )
        print(
            f"             send: total_bw={(avg_total_send / 1e9) / dispatch_total_s:.2f}  "
            f"nvl_bw={(avg_nvl_send / 1e9) / dispatch_total_s:.2f}  "
            f"rdma_bw={(avg_rdma_send / 1e9) / dispatch_total_s:.2f} GB/s"
        )
    print(
        f"Combine:     total={combine_avg * 1000:.2f} us "
        f"(min={combine_min * 1000:.2f}, max={combine_max * 1000:.2f})"
    )
    if combine_total_s > 0.0:
        print(
            f"             send: total_bw={(avg_total_recv / 1e9) / combine_total_s:.2f}  "
            f"nvl_bw={(avg_nvl_recv / 1e9) / combine_total_s:.2f}  "
            f"rdma_bw={(avg_rdma_recv / 1e9) / combine_total_s:.2f} GB/s"
        )
        print(
            f"             recv: total_bw={(avg_total_send / 1e9) / combine_total_s:.2f}  "
            f"nvl_bw={(avg_nvl_send / 1e9) / combine_total_s:.2f}  "
            f"rdma_bw={(avg_rdma_send / 1e9) / combine_total_s:.2f} GB/s"
        )
    print(
        f"Total (D+C): avg={total_avg * 1000:.2f} us, "
        f"min={total_min * 1000:.2f} us, max={total_max * 1000:.2f} us"
    )

    print("\n--- BW based on kernel time ---")
    print(f"Dispatch:    kernel={avg_dispatch_kernel:.2f} us")
    if dispatch_kernel_s > 0.0:
        print(
            f"             recv: total_bw={(avg_total_recv / 1e9) / dispatch_kernel_s:.2f}  "
            f"nvl_bw={(avg_nvl_recv / 1e9) / dispatch_kernel_s:.2f}  "
            f"rdma_bw={(avg_rdma_recv / 1e9) / dispatch_kernel_s:.2f} GB/s"
        )
        print(
            f"             send: total_bw={(avg_total_send / 1e9) / dispatch_kernel_s:.2f}  "
            f"nvl_bw={(avg_nvl_send / 1e9) / dispatch_kernel_s:.2f}  "
            f"rdma_bw={(avg_rdma_send / 1e9) / dispatch_kernel_s:.2f} GB/s"
        )
    print(f"Combine:     kernel={avg_combine_kernel:.2f} us")
    if combine_kernel_s > 0.0:
        print(
            f"             send: total_bw={(avg_total_recv / 1e9) / combine_kernel_s:.2f}  "
            f"nvl_bw={(avg_nvl_recv / 1e9) / combine_kernel_s:.2f}  "
            f"rdma_bw={(avg_rdma_recv / 1e9) / combine_kernel_s:.2f} GB/s"
        )
        print(
            f"             recv: total_bw={(avg_total_send / 1e9) / combine_kernel_s:.2f}  "
            f"nvl_bw={(avg_nvl_send / 1e9) / combine_kernel_s:.2f}  "
            f"rdma_bw={(avg_rdma_send / 1e9) / combine_kernel_s:.2f} GB/s"
        )
    print(f"Total (D+C): kernel={avg_dispatch_kernel + avg_combine_kernel:.2f} us")

    if dispatch_kernel_s > 0.0 or combine_kernel_s > 0.0:
        print(
            "\nByte counts (per rank avg): "
            f"total_send={avg_total_send / 1e6:.2f} MB ({avg_total_send_tokens} tokens), "
            f"rdma_send={avg_rdma_send / 1e6:.2f} MB ({avg_rdma_send_tokens} tokens), "
            f"rdma_recv={avg_rdma_recv / 1e6:.2f} MB ({avg_rdma_recv_tokens} tokens), "
            f"total_recv={avg_total_recv / 1e6:.2f} MB ({avg_total_recv_tokens} tokens)"
        )

    if avg_dispatch_kernel == 0.0 or avg_combine_kernel == 0.0:
        print("  NOTE: 0 us means no matching kernel was captured.")
        print("  Uncomment ktimer.dump() in ep_bench_core.cu to inspect captured kernel names.")


def print_setup_summary(results: list[dict[str, Any]]) -> None:
    group_times = [item["group_create_ms"] for item in results]
    handle_times = [item["handle_create_ms"] for item in results]
    print(f"\n=== Setup Timing (across {len(results)} ranks) ===")
    print(
        f"ncclEpCreateGroup:   avg={avg(group_times):.2f} ms, "
        f"min={min(group_times):.2f} ms, max={max(group_times):.2f} ms"
    )
    print(
        f"ncclEpCreateHandle:  avg={avg(handle_times):.2f} ms, "
        f"min={min(handle_times):.2f} ms, max={max(handle_times):.2f} ms"
    )


def print_validation_summary(results: list[dict[str, Any]]) -> None:
    dispatch_pass = all(item["dispatch_validation_pass"] for item in results)
    combine_pass = all(item["combine_validation_pass"] for item in results)
    max_diff = max(item["combine_validation_max_diff"] for item in results)
    print("\n=== Data Validation ===")
    print(f"Dispatch validation: {'PASSED' if dispatch_pass else 'FAILED'}")
    print(
        f"Combine validation:  {'PASSED' if combine_pass else 'FAILED'} "
        f"(calc_diff={max_diff:.6e})"
    )
    print(
        f"\nGlobal validation: Dispatch={'PASSED' if dispatch_pass else 'FAILED'}, "
        f"Combine={'PASSED' if combine_pass else 'FAILED'}"
    )


def main() -> int:
    args = build_parser().parse_args()
    rank, world_size, local_rank = require_torchrun_env()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run ep_bench.py")

    algorithm = normalize_algorithm(args.algorithm)
    if args.tokens < 0:
        raise RuntimeError("--tokens must be non-negative")
    if args.hidden <= 0 or args.top_k <= 0 or args.experts <= 0:
        raise RuntimeError("--hidden, --top-k, and --experts must be positive")
    if args.warmup < 0 or args.iters < 0:
        raise RuntimeError("--warmup and --iters must be non-negative")
    if rank == 0:
        print_configuration(args, algorithm, world_size)

    torch.cuda.set_device(local_rank)
    if args.disable_nvlink and algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"

    dist.init_process_group(backend="gloo")

    try:
        unique_id_bytes = get_unique_id_bytes(rank)
        core = load_core_library()

        bootstrap = EpBenchBootstrap(rank=rank, world_size=world_size, local_rank=local_rank)
        for i, value in enumerate(unique_id_bytes):
            bootstrap.nccl_unique_id[i] = value

        config = EpBenchConfig(
            algorithm=algorithm,
            num_tokens=max(args.tokens, 0),
            hidden=args.hidden,
            top_k=args.top_k,
            num_experts=args.experts,
            num_warmup=args.warmup,
            num_iters=args.iters,
            profile_mode=int(args.profile),
            disable_nvlink=int(args.disable_nvlink),
            use_fp8=int(args.use_fp8),
            validate_data=int(args.validate),
            dynamic_tokens=int(args.dynamic_tokens),
        )
        local_result = EpBenchLocalResult()
        rc = int(core.ep_bench_run(ctypes.byref(bootstrap), ctypes.byref(config), ctypes.byref(local_result)))

        gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result_to_dict(rank, local_result, rc))
        results = [item for item in gathered if item is not None]

        if rank == 0:
            errors = [item for item in results if item["rc"] != 0 or item["error_code"] != 0]
            if errors:
                print("Benchmark failed on one or more ranks:", file=sys.stderr)
                for item in sorted(errors, key=lambda entry: entry["rank"]):
                    message = item["error_message"] or "unknown error"
                    print(f"  rank {item['rank']}: {message}", file=sys.stderr)
                return 1

            print_local_lines(results, algorithm)
            if algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY:
                print_ll_summary(results)
            else:
                print_ht_summary(results)
            print_setup_summary(results)
            if args.validate:
                print_validation_summary(results)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
